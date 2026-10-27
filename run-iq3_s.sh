#!/bin/sh
# Strata launcher: qwen3.8-flash-next IQ3_S on the dual TU102 host, serving
# api.d07yx58.net's standing engine behind yarn nginx :8080 (plan/0149, call/0041).
#
# Context 524288 = 2 x the model's trained 262144 positions, extended with
# YaRN rope scaling, factor exactly 2 (524288/262144; the derivation lives
# here as the two integers so it can never drift):
#   TRAINED=262144  MAX_CONTEXT=524288  ROPE=MAX_CONTEXT/TRAINED=2
# The engine takes the factor as the decimal 2 (atof-parsed); strata's
# setup derives the same value for any context past 262144.
#
# Tuned trio from the 2026-10-19 calibration (pcie-frac 0.34,
# prefill-pipe 1536, spec-min-p 0.70). --expert-profile points at the
# SHIPPED profile in the worktree's data/ — required: a layer split across
# the two GPUs refuses to start without one (the engine names it).
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json --port 18195
