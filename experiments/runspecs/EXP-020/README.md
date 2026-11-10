# EXP-020 paired overall-head development trial

The approved card fixes two known inner splits, two heads and two model seeds.
Only six fresh full20 fits run; audited deep/seed7 inner results are immutable
references. No outer fitting, dense prediction or external evaluation occurs.

Local preparation (CPU initialization only; requires the existing exact K400
checkpoint, never downloads):

```text
.venv/Scripts/python.exe -B experiments/runspecs/EXP-020/initialization.py --observe-local-environment --out data/metadata/EXP-020/local-initialization-v1.json
.venv/Scripts/python.exe -B experiments/runspecs/EXP-020/prepare.py --initialization-proof data/metadata/EXP-020/cloud-initialization-v1.json
.venv/Scripts/python.exe -B experiments/tests/test_head_comparison.py
```

Local torch2.4.1 and original cloud torch2.5.1 do not produce the same complete
initial-state digest. Local four-model output is an observation, never a production
proof. First run this tiny CPU-only command in the unchanged original cloud
environment, with only the new model module/initializer and existing old code:

```text
python -B experiments/runspecs/EXP-020/initialization.py --out data/metadata/EXP-020/cloud-initialization-v1.json
```

It loads no bank or cache, constructs/releases one model at a time and never
forwards, downloads, creates an optimizer or uses GPU. It strictly requires
deep/seed7 to equal the historical c184 initialization and all common backbones
to agree. Copy the small proof back before local prepare; then copy the small
formal plan to the cloud. Production checks this proof and its frozen source SHA
before each fit, and checks each actual model state before creating Adam.

GPU command after code review, actual rental authorization and fee/timer setup:

```text
/bin/bash experiments/runspecs/EXP-020/run.sh --plan data/metadata/EXP-020/head-plan-v1 --out /root/exp020-results/head-s7s8-20261103-v1 --money-deadline-unix ACTUAL_AUTHORIZED_FEE_DEADLINE --platform-timer-receipt OPERATOR_VERIFIED_TIMER_RECEIPT
```

The fixed fit order is f01-linear-s7, f01-deep-s8, f01-linear-s8, then the same
three items for fold4. Each fit has a new Python process, fresh K400/model/Adam,
and the original frozen20 schedules and generator100007. Linear initialization
calls the complete old classifier before replacing its head; it never reseeds.
The original epoch execution block is byte-compared at prepare time. Existing
accepted inputs are reused; actual selected JPEG/RGB hashes are checked as read.
The two evolved governance documents are mapped explicitly to immutable old
snapshots. This is not a new full input acceptance or retrospective protocol edit.

The operator passes a real fee-protection deadline (total authorization20yuan),
not a predicted-completion stop. Forecasts only inform. The approved2yuan target
is not used as an automatic stopping threshold. A platform timer is independent
of the local watchdog; run.sh requests shutdown after success or technical failure.

Outputs may be on the system disk outside the input repository; local collection
maps them to data/results/exp020/RUN without rewriting original profile paths.
Each closed-item receipt lists actual relative paths, sizes and SHA, including
best/last optimizer-bearing checkpoints. closed-run.json closes final scientific
artifacts, excluding any later operator/shutdown logs. Collectors must not collect
an actively written fit. A partial run never receives a complete comparison.

comparison.json recomputes all selected raw logits into logit AUC and FP64 sigmoid
fixed0.5 recalls. Probability AUC is a separately named derivative; historical
files stay unchanged. All four linear items must individually satisfy AUC>=.75,
positive Recall>=.5 and negative Recall>=.5. Paired regressions remain visible.
Passing is only permission to consider the next design, not a video-level or
temporal-benefit claim. Dependencies match the existing torch/torchvision/numpy/
opencv EXP017 environment. No new runtime package is required.
