"""Current-user Windows DPAPI storage; never stores plaintext on disk.

No credential is read on import. Callers must enforce their execution and cost
authorization before calling load_runway_secret(). DPAPI is used without the
LOCAL_MACHINE flag, so this is tied to the current Windows user, not all users.
https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata
https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptunprotectdata
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
RUNWAY_KEY_PATH = ROOT / ".codex" / "secrets" / "runway.key.dpapi"
CRYPTPROTECT_UI_FORBIDDEN = 0x01


class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Local credential storage requires Windows current-user DPAPI")
    if not data:
        raise ValueError("Credential data is empty")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
                         ctypes.POINTER(DATA_BLOB), ctypes.c_void_p, ctypes.c_void_p,
                         wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
    function.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(data)
    source = DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = DATA_BLOB()
    try:
        description = None if decrypt else "Runway Dev API credential"
        if not function(ctypes.byref(source), description, None, None, None,
                        CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(output)):
            # Numeric Windows error only: never include credential material.
            raise OSError("Windows DPAPI operation failed (error %d)" % ctypes.get_last_error())
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.memset(ctypes.addressof(buffer), 0, ctypes.sizeof(buffer))
        if output.pbData:
            if decrypt:
                ctypes.memset(output.pbData, 0, output.cbData)
            kernel32.LocalFree(ctypes.cast(output.pbData, ctypes.c_void_p))


def validated_secret(value: str) -> str:
    value = value.strip()
    if not value or any(ord(character) < 32 for character in value):
        raise ValueError("API key must be nonempty and contain no control characters")
    return value


def store_runway_secret(value: str, path: Path = RUNWAY_KEY_PATH, *, replace=False) -> Path:
    path = Path(path).resolve()
    if path.exists() and not replace:
        raise FileExistsError("Credential already exists; use --replace explicitly to replace it")
    encrypted = _dpapi(validated_secret(value).encode("utf-8"), decrypt=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encrypted)  # Only DPAPI ciphertext is ever written.
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(str(temporary), str(path))
        else:
            # Windows rename refuses an existing target, including a concurrent write.
            os.rename(str(temporary), str(path))
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


def load_runway_secret(path: Path = RUNWAY_KEY_PATH) -> str:
    plaintext = _dpapi(Path(path).read_bytes(), decrypt=True)
    try:
        return validated_secret(plaintext.decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError("DPAPI credential is not valid UTF-8") from None
