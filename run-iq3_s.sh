#!/bin/sh
# Strata launcher: qwen3.8-flash-next IQ3_S on the dual TU102 host, serving
# api.d07yx58.net's standing engine behind yarn nginx :8080 (plan/0149, call/0041).
#
# Engine 0.1.52 (upstream v0.2.7 merged 2026-10-28): pipelined two-card serving
# via STRATA_PIPED_CONCURRENT + SPLIT_OWN + STAGE_TRIM in the config's env
# block (the #4285 "interactive column": a short chat next to a 250K+ prefill
# ends in ~19 s instead of sitting the whole prefill out).
#
# Context 524288 = 2 x the model's trained 262144 positions, extended with
# YaRN rope scaling, factor exactly 2 (524288/262144; the derivation lives
# here as the two integers so it can never drift):
#   TRAINED=262144  MAX_CONTEXT=524288  ROPE=MAX_CONTEXT/TRAINED=2
# The engine takes the factor as the decimal 2 (atof-parsed); strata's
# setup derives the same value for any context past 262144.
#
# Tuned trio from the 2026-10-19 calibration (pcie-frac 0.34,
# prefill-pipe 1536, spec-min-p 0.70; recalibrate after the engine bump if
# throughput shifts). --expert-profile is the SHIPPED data/ profile the
# dual-GPU layer split requires. Prefill auto:16384 per the community A/B
# (#4182: 16384 is the 4.2x sweet spot on 24 GB cards; 32768 regresses).
# Conversation cache 45056/6 = every slot holds a full 524288-token window.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json --port 18195
