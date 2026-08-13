#!/bin/bash
#
# entrypoint.serve.sh — start the rust harness as an HTTP server.
#
# Builds the harness argument list from the environment, so the compose file
# selects the checkpoint and the steering vector the way the llama services
# did. The interactive `harness` service keeps the bare binary as its
# entrypoint and takes its arguments on the command line; only the serving
# service needs this, because docker compose has no way to pass a flag
# conditionally.
#
# The four variables below are the ones the retired llama-cpu and llama-gpu
# entrypoints read, with the same empty-default-omits-the-flag shape. They
# were not renamed: interp/steering/wiring.py's steering_env emits exactly
# these names, and make steer-up passes them through, so a rename here would
# be a rename in four places to no effect.

set -euo pipefail

GREEN='\033[0;32m' YELLOW='\033[1;33m' RED='\033[0;31m' CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]]; then
    GREEN='' YELLOW='' RED='' CYAN='' NC=''
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

HARNESS_MODEL=${HARNESS_MODEL:-""}
HARNESS_PORT=${HARNESS_PORT:-8005}
CONTROL_VECTOR=${CONTROL_VECTOR:-""}
CONTROL_VECTOR_SCALE=${CONTROL_VECTOR_SCALE:-""}
CONTROL_VECTOR_LAYER_RANGE=${CONTROL_VECTOR_LAYER_RANGE:-""}
# Observation. The probe is scored in-band at every position of its layer and
# the log is where the per-request record lands; the log is what interp's
# probe monitor reads, and without it a probe's scores are computed and
# discarded, because serving has no end of run at which to report them.
HARNESS_PROBE=${HARNESS_PROBE:-""}
HARNESS_INTERP_LOG=${HARNESS_INTERP_LOG:-""}
# Gemma 4 checkpoints carry vision and audio towers the CPU backend does not
# implement, and refuses rather than half-loads. Serving is a text endpoint.
HARNESS_TEXT_ONLY=${HARNESS_TEXT_ONLY:-true}

if [[ -z "$HARNESS_MODEL" ]]; then
    print_error "HARNESS_MODEL: no checkpoint directory given"
    print_error "  Solution: set HARNESS_MODEL to a directory holding config.json and model.safetensors"
    exit 1
fi

CMD_ARGS=("--model" "$HARNESS_MODEL" "--serve" "$HARNESS_PORT")

if [[ "$HARNESS_TEXT_ONLY" == "true" ]]; then
    CMD_ARGS+=("--text-only")
fi

if [[ -n "$CONTROL_VECTOR" ]]; then
    if [[ ! -f "$CONTROL_VECTOR" ]]; then
        print_error "Control vector not found: $CONTROL_VECTOR"
        print_error "  Solution: check the vectors mount and the CONTROL_VECTOR path"
        exit 1
    fi
    CMD_ARGS+=("--control-vector" "$CONTROL_VECTOR")
    if [[ -n "$CONTROL_VECTOR_SCALE" ]]; then
        CMD_ARGS+=("--control-vector-scale" "$CONTROL_VECTOR_SCALE")
    fi
    # One argument holding two numbers, not two arguments: this is the string
    # steering_env already produces, and the harness parses it as a pair.
    if [[ -n "$CONTROL_VECTOR_LAYER_RANGE" ]]; then
        CMD_ARGS+=("--control-vector-layer-range" "$CONTROL_VECTOR_LAYER_RANGE")
    fi
fi

if [[ -n "$HARNESS_PROBE" ]]; then
    if [[ ! -f "$HARNESS_PROBE" ]]; then
        print_error "Probe not found: $HARNESS_PROBE"
        print_error "  Solution: check the probes mount and the HARNESS_PROBE path"
        exit 1
    fi
    CMD_ARGS+=("--probe" "$HARNESS_PROBE")
fi

if [[ -n "$HARNESS_INTERP_LOG" ]]; then
    CMD_ARGS+=("--interp-log" "$HARNESS_INTERP_LOG")
fi

print_info "Model: $HARNESS_MODEL"
print_info "Endpoint: 0.0.0.0:$HARNESS_PORT"
if [[ -n "$CONTROL_VECTOR" ]]; then
    print_info "Control vector: ${CONTROL_VECTOR}${CONTROL_VECTOR_SCALE:+ (scale ${CONTROL_VECTOR_SCALE})}${CONTROL_VECTOR_LAYER_RANGE:+ (blocks ${CONTROL_VECTOR_LAYER_RANGE})}"
else
    print_info "Control vector: none, serving unsteered"
fi
if [[ -n "$HARNESS_INTERP_LOG" ]]; then
    print_info "Interp log: ${HARNESS_INTERP_LOG}/harness${HARNESS_PROBE:+ (probe ${HARNESS_PROBE})}"
elif [[ -n "$HARNESS_PROBE" ]]; then
    print_warn "Probe set without HARNESS_INTERP_LOG: scores will be computed and discarded"
fi
# The CPU backend decodes at single-digit tokens per second on a 4B-class
# checkpoint, so a caller must pass a timeout matched to max_tokens rather
# than rely on a client default. See common/serving.py's harness entry.
print_warn "CPU backend: expect single-digit tokens per second; set client timeouts accordingly"
print_ok "Launch: /app/harness ${CMD_ARGS[*]}"

exec /app/harness "${CMD_ARGS[@]}"
