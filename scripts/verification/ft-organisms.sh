#!/usr/bin/env bash
# ft-organisms.sh: run FT.organisms for one organism recipe as one logged, resumable job.
#
# The entry (docs/verification/training/fine-tuning-loop.md, FT.organisms) is
# nine stages with three different serving states and about three hours of
# wall clock per organism, which is the wrong shape for a command block an
# operator pastes stage by stage. This script is that block made executable:
# every stage is the make target the entry names, in the entry's order, and
# the script brings the services each stage needs up and down itself. Run it
# through `make verify ENTRY=ft-organisms ARGS="--recipe ... --teacher ..."`,
# which tees the whole output to a log file; the log is the paste that
# closes the entry.
#
# A stage whose output already exists is skipped, so a failed run resumes
# where it stopped: the sentinels are the files the next stage reads. A
# stage that fails stops the script with its exit code; an invalid
# verification verdict (exit 1 from organism-verify) is the finding the
# entry exists to record, not something to retry here.
#
# Every path below is derived from the recipe and the tree (the staged base,
# the run and eval roots, the export root) through the same Python the
# pipeline uses, so no path is stated twice. The taboo recipe adds the tamper
# arm after the seal, on a copy, so the sealed organism FT.capture reads is
# untouched. Serving state on exit: everything down, whatever stage stopped.

set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The venv interpreter, as make/quality.mk resolves it (KB-95); the make
# target exports RUN_PY, and a direct invocation gets the workstation form.
RUN_PY="${RUN_PY-uv run --no-sync}"
# The reference recipe whose dataset locality is measured on: the organism's
# own corpus would measure the implant, not what it left alone.
REFERENCE_RECIPE="training/recipes/reference/sft-qlora-e2b.yaml"
LMEVAL_LIMIT=20
# The compose file's start_period for a cold 10 GB load, not the registry's
# 60 s default.
HARNESS_BUDGET_S=600
TAMPER_DIR="/tmp/ft7-tamper"

usage() {
    echo "usage: $0 --recipe training/recipes/organisms/<name>.yaml --teacher <model path>"
    echo "  --teacher is the model vllm-gpu serves for generation, host or container spelling;"
    echo "  it is required until the generation stage has completed."
}

RECIPE="" TEACHER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe) RECIPE="$2"; shift 2 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) print_error "unknown argument: $1"; usage; exit 2 ;;
    esac
done
if [[ -z "$RECIPE" ]]; then
    print_error "--recipe is required"
    usage
    exit 2
fi
if [[ ! -f "$RECIPE" ]]; then
    print_error "recipe not found: $RECIPE"
    exit 2
fi

# Everything the stages need to name, read once from the recipe and the
# tree. Printed as shell assignments and eval'd; a base that is not staged
# is refused here rather than by the trainer an hour in.
facts() {
    # shellcheck disable=SC2016  # the $ signs are Python's, not the shell's
    $RUN_PY python - "$RECIPE" "$REFERENCE_RECIPE" <<'PY'
import os
import shlex
import sys

from common import serving
from training.eval.runner import DEFAULT_RUN_ROOT
from training.models.staged import resolve_model_source
from training.recipes.loader import load_recipe

recipe = load_recipe(sys.argv[1])
reference = load_recipe(sys.argv[2])
if not recipe.organism.enabled:
    sys.exit(f"{sys.argv[1]}: organism.enabled is false; not an organism recipe")
base = resolve_model_source(recipe.model.hf.id)
if not os.path.isfile(os.path.join(base, "config.json")):
    sys.exit(f"base {recipe.model.hf.id} is not staged; run make train-download first")
name = recipe.run.name
export_host = os.path.join(serving.HOST_MODELS, "trained", name, "hf")
values = {
    "NAME": name,
    "KIND": recipe.organism.kind,
    "OUT_DIR": recipe.organism.out_dir,
    "DATASET": recipe.sft.dataset,
    "BASE": base,
    "CHECKPOINT": os.path.join(
        recipe.run.output_root,
        name,
        "checkpoints",
        f"checkpoint-{recipe.schedule.total_steps}",
    ),
    "EXPORT_HOST": export_host,
    "EXPORT_CONTAINER": serving.model_paths(export_host)[0],
    "EVAL_DIR": os.path.join(DEFAULT_RUN_ROOT, f"ft7-{name}"),
    "LOCALITY_DATASET": reference.sft.dataset,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

echo "=== FT.organisms: $RECIPE ==="
echo
print_run "Reading the recipe and the tree"
eval "$(facts)"
print_info "Organism: $NAME ($KIND) under $OUT_DIR"
print_info "Base: $BASE"
print_info "Export: $EXPORT_HOST"
print_info "Eval records: $EVAL_DIR"
echo

STAGES=9
stage_run()  { print_run "$1/$STAGES $2"; }
stage_ok()   { print_ok "$2: exit 0"; echo; }
stage_skip() { print_skip "$2: $3 exists"; echo; }

# Serving state is released by the stage that took it, and by the trap when
# a stage fails, so a failed verify never leaves the harness and the judge
# running for the next GPU consumer to find.
SERVING_UP=""
teardown() {
    if [[ "$SERVING_UP" == "verify" ]]; then
        print_run "Bringing the harness and the judge down after a failure"
        make harness-serve-down || true
        make down || true
    elif [[ "$SERVING_UP" == "generate" ]]; then
        print_run "Bringing the teacher down after a failure"
        make down || true
    fi
}
trap teardown EXIT

# 1. Generation: ground truth drawn, the corpus written through the teacher
#    on vllm-gpu, then blinded. generate-teacher swaps vllm-gpu to the
#    teacher and restores the default model afterwards; gpu-up first gives
#    the restore something to restore.
stage_run 1 "generate"
if [[ -f "$OUT_DIR/data-blind/manifest.json" ]]; then
    stage_skip 1 "generate" "$OUT_DIR/data-blind/manifest.json"
else
    if [[ -z "$TEACHER" ]]; then
        print_error "--teacher is required for the generation stage"
        echo "  Solution: pass the model vllm-gpu serves, host or container spelling"
        exit 2
    fi
    SERVING_UP="generate"
    make gpu-up
    make organism-generate RECIPE="$RECIPE" TEACHER="$TEACHER"
    make down
    SERVING_UP=""
    stage_ok 1 "generate"
fi

# 2. prepare-sft over the blind corpus. --source is the JSONL, not its
#    directory: the verb opens the path.
stage_run 2 "prepare-sft"
if [[ -f "$DATASET/manifest.json" ]]; then
    stage_skip 2 "prepare-sft" "$DATASET/manifest.json"
else
    make prepare-sft ARGS="--source $OUT_DIR/data-blind/data.jsonl \
        --tokenizer $BASE --model-type gemma4 --out $DATASET"
    stage_ok 2 "prepare-sft"
fi

# 3. Training, inference down. The recipe's total_steps names the checkpoint
#    the export reads.
stage_run 3 "train-sft"
if [[ -f "$CHECKPOINT/meta.json" ]]; then
    stage_skip 3 "train-sft" "$CHECKPOINT/meta.json"
else
    make train-sft CONFIG="$RECIPE"
    stage_ok 3 "train-sft"
fi
ls "$(dirname "$CHECKPOINT")"
echo

# 4. Export the merged directory the harness serves.
stage_run 4 "export-hf"
if [[ -f "$EXPORT_HOST/model.safetensors" ]]; then
    stage_skip 4 "export-hf" "$EXPORT_HOST/model.safetensors"
else
    make export-hf CHECKPOINT="$CHECKPOINT" ARGS="--name $NAME"
    stage_ok 4 "export-hf"
fi
grep -E '"(model_hash|hf_dir)"' "$(dirname "$EXPORT_HOST")/manifest.json"
echo

# 5. eval-ft on base and organism, inference down, into a fixed directory so
#    the records have known names; eval-ft resumes per record on its own.
stage_run 5 "eval-ft"
if [[ -f "$EVAL_DIR/lmeval_tuned.json" ]]; then
    stage_skip 5 "eval-ft" "$EVAL_DIR/lmeval_tuned.json"
else
    make eval-ft ARGS="--base $BASE --tuned $EXPORT_HOST --evaluators holdout,lmeval \
        --dataset $LOCALITY_DATASET --lmeval-limit $LMEVAL_LIMIT --out-dir $EVAL_DIR"
    stage_ok 5 "eval-ft"
fi
ls "$EVAL_DIR"
echo

# 6. Verification: the harness serving the export (a CPU consumer) beside
#    vllm-gpu as the judge. An invalid verdict exits 1 and stops here.
stage_run 6 "organism-verify"
if [[ -f "$OUT_DIR/verification.json" ]]; then
    stage_skip 6 "organism-verify" "$OUT_DIR/verification.json"
else
    SERVING_UP="verify"
    make gpu-up
    HARNESS_MODEL="$EXPORT_CONTAINER" make harness-serve
    $RUN_PY python -c "from common import serving; print(serving.wait_healthy(serving.SERVICES['harness'], $HARNESS_BUDGET_S))"
    make organism-verify RECIPE="$RECIPE" \
        BASE_METRICS="$EVAL_DIR/holdout_base.json $EVAL_DIR/lmeval_base.json" \
        SUBJECT_METRICS="$EVAL_DIR/holdout_tuned.json $EVAL_DIR/lmeval_tuned.json"
    make harness-serve-down
    make down
    SERVING_UP=""
    stage_ok 6 "organism-verify"
fi

# 7. Seal: refuses an invalid verdict, records the export and its model_hash.
stage_run 7 "organism-seal"
if [[ -f "$OUT_DIR/ground_truth.json" ]]; then
    stage_skip 7 "organism-seal" "$OUT_DIR/ground_truth.json"
else
    make organism-seal RECIPE="$RECIPE" ARGS="--checkpoint $EXPORT_HOST"
    stage_ok 7 "organism-seal"
fi

# 8. check-seal on the sealed directory, and the two manifest fields the
#    entry judges.
stage_run 8 "organism-check-seal"
make organism-check-seal DIR="$OUT_DIR"
grep -E '"(checkpoint|model_hash)"' "$OUT_DIR/manifest.json"
stage_ok 8 "organism-check-seal"

# 9. The tamper arm, taboo only: a copy with one byte appended to the ground
#    truth must be refused, and the untouched directory must still pass.
stage_run 9 "tamper arm"
if [[ "$KIND" != "taboo" ]]; then
    print_skip "tamper arm: the $KIND organism has no tamper arm"
    echo
else
    rm -rf "$TAMPER_DIR" && cp -r "$OUT_DIR" "$TAMPER_DIR"
    echo >> "$TAMPER_DIR/ground_truth.json"
    if make organism-check-seal DIR="$TAMPER_DIR"; then
        print_error "tamper arm: check-seal passed on the tampered copy; the seal is broken"
        exit 1
    fi
    print_ok "tamper arm: check-seal refused the tampered copy (exit 1)"
    make organism-check-seal DIR="$OUT_DIR"
    stage_ok 9 "tamper arm"
fi

print_ok "ft-organisms: $NAME complete"
