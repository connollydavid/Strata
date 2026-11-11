#!/bin/sh
# Strata launcher: qwen3.8-flash-next Unsloth Dynamic v3.1 UD-IQ4_XS with the
# Q8_0 engram, on the dual TU102 host behind yarn nginx :8080 (plan/0154, the
# single big-bang switchover: engine, model, table, and cache all change in
# one window).
#
# Engine 0.1.59 (upstream v0.2.13 merged 2026-11-11). The set's four files:
# shards 1-3 are the model (64.87 GB; the bundled IQ4_NL table shard is
# excluded at download), shard 4 is the Q8_0 engram table stamped 4-of-4 by
# unsloth's set_split.py (54.4 GB, 8.5 bpw, table RMSE vs BF16 0.0053
# against the IQ4_NL rung's 0.0757). The engine's row reader takes Q8_0
# tables natively (split_engine.py PLE_TYPES).
#
# Q5_0 experts: UD-IQ4_XS-v31 carries Q5_0 in 33 layers; upstream ships the
# sm_75 kernels built but never ran them (no Turing card). Our window runs
# the GPU parity check before this line is called healthy.
#
# Context 524288 = 2 x the trained 262144 positions, YaRN factor exactly 2
# (the derivation lives here as the two integers so it can never drift):
#   TRAINED=262144  MAX_CONTEXT=524288  ROPE=MAX_CONTEXT/TRAINED=2
#
# Conversation cache 30208/4: the big-bang RAM concession. Every slot still
# holds a full 524288-token window (7552 MiB a slot, just above the IQ3_S
# line's 7509), and the ~15 GiB the pool gives back funds the Q8_0 table's
# page cache (54.4 GB table against the IQ4_NL's 28.8; ple rows are 170 B
# at Q8_0 against 90 at IQ4_NL, so --ple-row-cache's byte cost roughly
# doubles at the same row count - watch it in the soak).
#
# The tuned trio (pcie-frac 0.34, prefill-pipe 1536, spec-min-p 0.70) was
# calibrated for the IQ3_S set on engine 0.1.52 (2026-10-19); recalibrate
# after the switchover if throughput shifts. --expert-profile keeps the
# SHIPPED data/ profile: the docs' same-model rule (the profile is a usage
# fingerprint, not a quant artifact) and the dual-GPU layer split requires
# one. Pictures carry over from plan/0153 unchanged: --vision with the
# reader pinned to GPU1 and --vram-reserve-mib 700 budgeting it.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python serve/server.py --engine strata --config strata-ud_v31_iq4xs.json --port 18195
