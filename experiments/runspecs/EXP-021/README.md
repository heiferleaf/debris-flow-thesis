# EXP-021 fixed oriented-energy development contrast

Current deployment uses `energy-plan-v2`. Its numeric schedules and kernels are
unchanged from v1. The first real CPU entry check stopped at `bash -n` because
Windows text output had changed run.sh to CRLF; zero training updates occurred.
The shell is now byte-preserved LF. v1 plans/manifests and the before-bytes remain
immutable; v2 only records this operational revision plus the updated shell/README
source fingerprints. No A weights or mathematical checks were repeated for it.

This approved card adds exactly112 zero-initialized weights to the existing
linear RGB classifier. B/C share centered input, kernels, epsilon, dimensions,
RGB initialization, schedules and training recipe. Only slice squaring versus
square-after-sum differs. This is a known-split development trial, not independent
testing, unseen-camera generalization or a physical velocity estimator.

```text
.venv/Scripts/python.exe -B experiments/tests/test_oriented_energy.py
.venv/Scripts/python.exe -B experiments/runspecs/EXP-021/prepare.py
/root/miniconda3/bin/python -B experiments/runspecs/EXP-021/pipeline.py --help
```

Preparation is local metadata/SHA only. It inherits actual prior pixel acceptance
and explicit historical/current protocol mapping, without re-decoding video or
rechecking all JPEG pixels. Kernels are frozen in EXP-021/kernels-v1; all108 share
one scale and both arms use the same fixed epsilon. Stable FP64 intermediate
centering computes one scalar per clip then casts back to the original dtype;
the RGB backbone path stays FP32 and unchanged. There is no post-centering clamp.
Constant input has zero energy and stable-fraction1 in each region, not an allzero
112-vector. The separable basis is checked against direct FP64 kernels/slices.

After code review and actual rental/fee-timer authorization only:

```text
/bin/bash experiments/runspecs/EXP-021/run.sh --plan data/metadata/EXP-021/energy-plan-v2 --out /root/exp021-results/energy-s7-20261103-v1 --reference-root /root/exp020-results/head-s7s8-20261103-v2 --ffmpeg ORIGINAL_CHECKED_FFMPEG7 --money-deadline-unix ACTUAL_TOTAL_AUTHORIZATION_FEE_DEADLINE --platform-timer-receipt ACTUAL_OPERATOR_VERIFIED_TIMER_RECEIPT
```

The plan records local A root=data/results/exp020/head-s7s8-20261103-v2 and cloud
A root=/root/exp020-results/head-s7s8-20261103-v2. Source paths in old profiles are
not rewritten; no A weights are copied or symlinked. The production RGB component
must match f3b66... at each before-Adam check. Local torch2.4 fingerprints are
never substituted for the cloud's existing torch2.5 proof. New parameters/buffers
do not consume RNG, and B/C complete initial state and CPU RNG are cross-checked.

Stages: f01-B-s7, f01-C-s7, f04-B-s7, f04-C-s7, then dense-f01-A/B/C and
dense-f04-A/B/C. Each fit is a separate full20 process with generator100007;
the original ordinary-BCE epoch block is unchanged. B/C energy weights belong
to the1e-3 head group. There is no score-based fit cancellation or extra seed.

Per inner role, A/B/C use one actual canonical dense decode. B/C share descriptions
of each same actual batch. There is no cross-source descriptor cache: supervised
JPEG features are never reused for dense decoded input merely by physical index.
All video scores remain provisional until the old canonical RGB112 digest closes
at EOF. No dense score selects E. The frozen23/24 roles share5 videos (42 unique).

closed-STAGE.json lists relative path/size/SHA with fit_closed or dense_closed;
closed-run.json closes all finalized artifacts. Collect only these closed files.
Later shutdown-receipt.json is operational evidence excluded from the scientific
closure. The shell uses the actual absolute Python path and a bounded shutdown
trap; a platform fee timer remains independently required. Forecasts are advisory,
the4yuan target is not a forecast stopping rule, and actual total20yuan remains
hard-protected by the operator-supplied money deadline.

comparison.json retains all clip floors, per-role confusion, unique R/W and
role-inconsistent changes, and session FPR (roles per video first, videos per
session second, sessions equally for macro). Every required gate must pass to
consider a formal matched A/C validation design. Failure does not trigger new
training, threshold/kernel/epsilon/region search or external evaluation.
