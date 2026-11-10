"""Experimental depth-averaged water solver; not site-calibrated hydraulics.

Hydrostatic reconstruction/Rusanov finite volumes, Manning drag and horizontal
momentum diffusion. Reference: Audusse et al. 2004, DOI 10.1137/S1064827503431090.
"""
import numpy as np

G = 9.81


def face(left, right, zl, zr, ml, mr, normal, viscosity, dx):
    left, right, zl, zr = left.copy(), right.copy(), zl.copy(), zr.copy()
    wall_right, wall_left = ml & ~mr, mr & ~ml
    right[wall_right] = left[wall_right]
    right[..., normal][wall_right] *= -1
    zr[wall_right] = zl[wall_right]
    left[wall_left] = right[wall_left]
    left[..., normal][wall_left] *= -1
    zl[wall_left] = zr[wall_left]
    hl, hr = left[..., 0], right[..., 0]
    vl = left[..., 1:] / np.maximum(hl[..., None], 1e-10)
    vr = right[..., 1:] / np.maximum(hr[..., None], 1e-10)
    crest = np.maximum(zl, zr)
    a, b = np.maximum(hl + zl - crest, 0), np.maximum(hr + zr - crest, 0)
    ul = np.concatenate([a[..., None], a[..., None] * vl], -1)
    ur = np.concatenate([b[..., None], b[..., None] * vr], -1)
    fl, fr = ul * vl[..., normal - 1, None], ur * vr[..., normal - 1, None]
    fl[..., normal] += .5 * G * a * a
    fr[..., normal] += .5 * G * b * b
    wave = np.maximum(np.abs(vl[..., normal - 1]) + np.sqrt(G * a),
                      np.abs(vr[..., normal - 1]) + np.sqrt(G * b))
    common = .5 * (fl + fr) - .5 * wave[..., None] * (ur - ul)
    if viscosity:
        # Viscous ghost velocities impose zero tangential velocity at the wall
        # face, not zero velocity in an entire bank strip.
        vr[wall_right] = -vl[wall_right]
        vl[wall_left] = -vr[wall_left]
        common[..., 1:] -= viscosity * .5 * (hl + hr)[..., None] * (vr - vl) / dx
    common[~(ml | mr)] = 0
    lf, rf = common.copy(), common.copy()
    lf[..., normal] += .5 * G * (hl * hl - a * a)
    rf[..., normal] += .5 * G * (hr * hr - b * b)
    return lf, rf, common[..., 0]


def extend(state, bed, wet, boundary, t):
    u = np.pad(state, ((1, 1), (1, 1), (0, 0)), mode='edge')
    z = np.pad(bed, ((1, 1), (1, 1)), mode='edge')
    m = np.pad(wet, 1, constant_values=False)
    if boundary['kind'] == 'periodic_x':
        u[1:-1, 0] = state[:, -1]; u[1:-1, -1] = state[:, 0]
        z[1:-1, 0] = bed[:, -1]; z[1:-1, -1] = bed[:, 0]
        m[1:-1, 0] = wet[:, -1]; m[1:-1, -1] = wet[:, 0]
    elif boundary['kind'] == 'channel_y':
        m[0, 1:-1] = wet[0]; m[-1, 1:-1] = wet[-1]
        h = np.maximum(state[0, :, 0], .02)
        v = state[0, :, 2] / h
        outgoing = v - 2 * np.sqrt(G * h)
        q = boundary['inflow_per_width'] * (1 + .025 * np.sin(2 * np.pi * t / 7)
                                             + .015 * np.sin(2 * np.pi * t / 2.3))
        ghost_h = h.copy()
        for _ in range(8):
            residual = q / ghost_h - 2 * np.sqrt(G * ghost_h) - outgoing
            derivative = -q / ghost_h**2 - np.sqrt(G / ghost_h)
            ghost_h = np.clip(ghost_h - residual / derivative, .02, 3)
        u[0, 1:-1, 0] = ghost_h
        u[0, 1:-1, 1] = 0
        u[0, 1:-1, 2] = q
        target = np.maximum(boundary['outlet_stage'] - bed[-1], .02)
        last_h = np.maximum(state[-1, :, 0], .02)
        plus = state[-1, :, 2] / last_h + 2 * np.sqrt(G * last_h)
        outlet_v = plus - 2 * np.sqrt(G * target)
        u[-1, 1:-1, 0] = target
        u[-1, 1:-1, 1] = target * state[-1, :, 1] / last_h
        u[-1, 1:-1, 2] = target * outlet_v
    return u, z, m


def advance(state, bed, wet, dx, dt, boundary, t, manning=.035, viscosity=.03, force_x=0):
    u, z, m = extend(state, bed, wet, boundary, t)
    xl, xr, xm = face(u[1:-1, :-1], u[1:-1, 1:], z[1:-1, :-1], z[1:-1, 1:],
                      m[1:-1, :-1], m[1:-1, 1:], 1, viscosity, dx)
    yl, yr, ym = face(u[:-1, 1:-1], u[1:, 1:-1], z[:-1, 1:-1], z[1:, 1:-1],
                      m[:-1, 1:-1], m[1:, 1:-1], 2, viscosity, dx)
    new = state - dt / dx * (xl[:, 1:] - xr[:, :-1] + yl[1:] - yr[:-1])
    new[~wet] = 0
    if new[..., 0][wet].min() < -1e-9 or not np.isfinite(new).all():
        raise ValueError('Unstable or negative-depth state')
    new[..., 1] += dt * force_x * new[..., 0]
    if manning:
        h = np.maximum(new[..., 0], 1e-8)
        speed = np.linalg.norm(new[..., 1:], axis=-1) / h
        drag = G * manning**2 * speed / h**(4 / 3)
        new[..., 1:] /= (1 + dt * drag)[..., None]
    net = dt * dx * (xm[:, 0].sum() - xm[:, -1].sum() + ym[0].sum() - ym[-1].sum())
    return new, float(net)


def timestep(state, wet, dx, viscosity):
    h = np.maximum(state[..., 0][wet], 1e-8)
    v = state[..., 1:][wet] / h[:, None]
    c = np.sqrt(G * h)
    dt = .28 * dx / (np.max(np.abs(v[:, 0]) + c) + np.max(np.abs(v[:, 1]) + c))
    return min(dt, .18 * dx * dx / viscosity) if viscosity else dt


def verify():
    ny, nx, dx = 24, 64, .25
    yy, xx = np.mgrid[:ny, :nx] * dx
    wet = np.ones((ny, nx), bool)
    bed = .2 * np.exp(-((xx - 8)**2 + (yy - 3)**2) / 3)
    state = np.zeros((ny, nx, 3)); state[..., 0] = .8 - bed
    initial = state[..., 0].sum()
    for i in range(100):
        state, _ = advance(state, bed, wet, dx, .01, {'kind': 'closed'}, i * .01, 0, 0)
    speed = float(np.linalg.norm(state[..., 1:] / state[..., :1], axis=-1).max())
    assert speed < 1e-10
    mass = float(abs(state[..., 0].sum() - initial) / initial)
    assert mass < 1e-10
    state[:] = 0; state[..., 0] = .7; state[..., 1] = .84
    for i in range(100):
        state, _ = advance(state, np.zeros_like(bed), wet, dx, .01, {'kind': 'periodic_x'}, i * .01, .035, 0)
    exact = 1.2 / (1 + G * .035**2 / .7**(4 / 3) * 1.2)
    error = float(np.abs(state[..., 1] / state[..., 0] - exact).max())
    assert error < 1e-9
    return dict(still_water_max_speed=speed, closed_mass_relative_error=mass, uniform_drag_max_error=error)
