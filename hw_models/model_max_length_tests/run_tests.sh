#!/bin/bash
#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################
#
# Runs the max-context suite one test at a time, each in its own pytest process, and
# writes every test's output to its own log file.
#
# Why not just `pytest .`? Building a second engine inside one process means tearing
# the first one down, and vLLM's worker processes do not always go away when that
# happens -- the next engine then contends with the previous one for the devices. One
# process per test sidesteps it entirely: the process exits, and whatever it left
# behind is reaped before the next test starts.
#
# Reaping is done by process group, not by name. Each pytest runs in its own group
# (bash job control gives a background job its own pgid), so killing the group can
# only ever hit that pytest and its descendants -- never anything else you have
# running on the machine. --reap-strays adds an opt-in, deliberately narrow sweep for
# vLLM processes that escaped their group.
#
#   ./run_tests.sh                          # every collected test, one at a time
#   ./run_tests.sh --group-by-engine        # batch cases that share an engine (faster)
#   ./run_tests.sh -k selftest              # only the device-free self-tests
#   ./run_tests.sh --tier full --tp 8       # full sweep to the ceiling
#   ./run_tests.sh --list                   # show what would run, then exit
#
# Exit status is 0 only if every test passed or skipped.

set -uo pipefail  # deliberately not -e: a failing test must not end the run

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
TEST_FILES=(test_token_math_selftest.py test_ctx_sweep.py test_ctx_boundary.py)
QAIC_UTIL="/opt/qti-aic/tools/qaic-util"

# --- defaults ---------------------------------------------------------------
OUTDIR=""
TIMEOUT=3600        # per test, seconds
GRACE=20            # seconds to wait for a TERMed process group before KILL
SETTLE=10           # seconds between tests, for the devices to be released
MODE="per-test"     # or "group-by-engine"
KFILTER=""
STOP_ON_FAIL=0
REAP_STRAYS=0
LIST_ONLY=0
DRY_RUN=0
PYTEST_EXTRA=()
TIER_OVERRIDE=""
TP_OVERRIDE=""
MODELS_OVERRIDE=""
BATCH_OVERRIDE=""
MODALITIES_OVERRIDE=""
PL_OVERRIDE=""
GL_OVERRIDE=""
DEVICE_IDS_OVERRIDE=""
RERUN_FAILED_OVERRIDE=""

usage() {
  cat << EOM
Usage: run_tests.sh [options] [-- <extra pytest args>]

Models, TP sizes, context targets, batch sizes and every assertion/device threshold
live in suite_config.json -- edit that file for anything not covered by a flag below.

  --group-by-engine   One pytest process per engine instead of per test. Cases that
                      share an engine (all modalities at one context point; the three
                      boundary tests) run together, so the engine is built once. No
                      process ever builds a second engine, which is the condition that
                      causes the crash. Much faster; slightly less isolated.
  --per-test          One pytest process per test (default).
  -k EXPR             Passed to pytest collection, e.g. -k selftest, -k 8192.
  --tier TIER         smoke (default) or full. Patches sweep.tier for this run only.
                      ("custom" also exists but is set by --pl/--gl, not directly.)
  --tp LIST           Comma-separated TP sizes. Patches selection.tp_sizes.
  --models LIST       Comma-separated model ids. Patches selection.models.
  --batch LIST        Comma-separated batch sizes. Patches selection.batch_sizes.
                      Each distinct batch size is a distinct engine (the KV budget
                      covers that many full-length sequences), so this multiplies
                      engine builds. Needs the paged-attention decode path, i.e. do
                      not also set QAIC_SDPA_DECODE=1.
  --modalities LIST   Comma-separated modality names (e.g. text,image_single).
                      Patches selection.modalities. A model that does not support a
                      listed modality simply contributes no cases for it. Omit for
                      every modality each selected model supports (the default).
  --pl PROMPT_LEN     Run exactly this one (prompt_len, gen_len) point -- must be
  --gl GEN_LEN        given together with --gl. Forces sweep.tier to "custom" and
                      patches sweep.custom_targets to this single pair, so no
                      smoke/full auto-ceiling point is added alongside it. Combine
                      with --models/--tp/--modalities/--batch for "run exactly this
                      model/TP/modality/batch at this prompt+gen length end to end,"
                      e.g.: ./run_tests.sh --models Qwen/Qwen3-32B --tp 8 --modalities text --pl 8192 --gl 64
  -o, --outdir DIR    Log directory (default ciLogs_vlm_max_context_<timestamp>).
  --timeout SECONDS   Per-test timeout (default ${TIMEOUT}).
  --settle SECONDS    Pause between tests (default ${SETTLE}).
  --grace SECONDS     Wait after TERM before KILL (default ${GRACE}).
  --stop-on-fail      Stop at the first failure instead of running everything.
  --reap-strays       Also kill user-owned vLLM processes that escaped their process
                      group. Narrow by design: only processes started after this
                      script, and only ones whose command line looks like a vLLM
                      worker. Off by default because it matches by name. Cannot be
                      combined with --device-ids (see below).
  --device-ids POOL   Run concurrently across this device-ID pool instead of one
                      batch at a time. Accepts a range ("48-63"), a comma list
                      ("48,50,55"), or a mix ("48,50-55,60"). Each batch (a single
                      test in --per-test mode, a shared-engine group in
                      --group-by-engine mode) claims a disjoint slice sized to its
                      TP the moment enough devices are free, and the next queued
                      batch claims a slice the instant one is released -- e.g. 16
                      devices and several TP=8 batches run two at a time. Devices
                      get a --settle cooldown after release before reuse. Dispatch,
                      per-job timeout/kill, and the final summary are handled by
                      scheduler.py; incompatible with --reap-strays (a name/age
                      stray-sweep cannot safely tell which concurrent job an
                      escaped process belongs to).
  --rerun-failed PATH Re-run only the FAIL/TIMEOUT/ERROR/NOT-RUN entries from an
                      earlier run's summary.txt, instead of the normal
                      suite_config.json-driven collection. PATH is that run's log
                      directory (its summary.txt is used) or a summary.txt path
                      directly. A --per-test entry is already a real pytest node
                      id; a --group-by-engine batch name is expanded back to its
                      member node ids via that run's jobs.manifest (written
                      whenever --device-ids was used -- a plain --group-by-engine
                      run without --device-ids has no other record of batch
                      membership to expand from). Replaces collection entirely,
                      so it cannot be combined with -k or any of
                      --tier/--tp/--models/--batch/--modalities/--pl/--gl.
  --list              Print the tests that would run, then exit.
  --dry-run           Print the pytest commands without running them.
  -h, --help          This message.

Anything after -- is appended to every pytest invocation.
EOM
}

# --- argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --group-by-engine) MODE="group-by-engine"; shift ;;
    --per-test)        MODE="per-test"; shift ;;
    -k)                KFILTER="${2:?-k needs an expression}"; shift 2 ;;
    --tier)            TIER_OVERRIDE="${2:?--tier needs a value}"; shift 2 ;;
    --tp)              TP_OVERRIDE="${2:?--tp needs a value}"; shift 2 ;;
    --models)          MODELS_OVERRIDE="${2:?--models needs a value}"; shift 2 ;;
    --batch)           BATCH_OVERRIDE="${2:?--batch needs a value}"; shift 2 ;;
    --modalities)      MODALITIES_OVERRIDE="${2:?--modalities needs a value}"; shift 2 ;;
    --pl)              PL_OVERRIDE="${2:?--pl needs a prompt_len}"; shift 2 ;;
    --gl)              GL_OVERRIDE="${2:?--gl needs a gen_len}"; shift 2 ;;
    -o|--outdir)       OUTDIR="${2:?--outdir needs a path}"; shift 2 ;;
    --timeout)         TIMEOUT="${2:?--timeout needs seconds}"; shift 2 ;;
    --settle)          SETTLE="${2:?--settle needs seconds}"; shift 2 ;;
    --grace)           GRACE="${2:?--grace needs seconds}"; shift 2 ;;
    --stop-on-fail)    STOP_ON_FAIL=1; shift ;;
    --reap-strays)     REAP_STRAYS=1; shift ;;
    --device-ids)      DEVICE_IDS_OVERRIDE="${2:?--device-ids needs a pool, e.g. 48-63}"; shift 2 ;;
    --rerun-failed)    RERUN_FAILED_OVERRIDE="${2:?--rerun-failed needs a run directory or summary.txt path}"; shift 2 ;;
    --list)            LIST_ONLY=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    --)                shift; PYTEST_EXTRA=("$@"); break ;;
    *) echo "Invalid option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
done

cd "$SCRIPT_DIR" || exit 2

if [ -n "$PL_OVERRIDE" ] || [ -n "$GL_OVERRIDE" ]; then
  if [ -z "$PL_OVERRIDE" ] || [ -z "$GL_OVERRIDE" ]; then
    echo "--pl and --gl must be given together (one explicit prompt_len + gen_len point)" >&2
    exit 2
  fi
  if ! [[ "$PL_OVERRIDE" =~ ^[0-9]+$ ]] || ! [[ "$GL_OVERRIDE" =~ ^[0-9]+$ ]]; then
    echo "--pl and --gl each take a single non-negative integer (got --pl $PL_OVERRIDE --gl $GL_OVERRIDE)" >&2
    exit 2
  fi
fi

if [ -n "$DEVICE_IDS_OVERRIDE" ] && [ "$REAP_STRAYS" -eq 1 ]; then
  echo "--device-ids and --reap-strays cannot be combined: a name/age stray-sweep" >&2
  echo "cannot safely tell which concurrent job an escaped process belongs to." >&2
  exit 2
fi

if [ -n "$RERUN_FAILED_OVERRIDE" ]; then
  if [ -n "$TIER_OVERRIDE" ] || [ -n "$TP_OVERRIDE" ] || [ -n "$MODELS_OVERRIDE" ] ||
     [ -n "$BATCH_OVERRIDE" ] || [ -n "$MODALITIES_OVERRIDE" ] || [ -n "$PL_OVERRIDE" ] ||
     [ -n "$GL_OVERRIDE" ] || [ -n "$KFILTER" ]; then
    echo "--rerun-failed replaces collection entirely, so it cannot be combined with" >&2
    echo "-k or --tier/--tp/--models/--batch/--modalities/--pl/--gl (which only affect" >&2
    echo "the normal suite_config.json-driven collection this bypasses)." >&2
    exit 2
  fi
  if [ -d "$RERUN_FAILED_OVERRIDE" ]; then
    RERUN_SUMMARY_FILE="$RERUN_FAILED_OVERRIDE/summary.txt"
  else
    RERUN_SUMMARY_FILE="$RERUN_FAILED_OVERRIDE"
  fi
  if [ ! -f "$RERUN_SUMMARY_FILE" ]; then
    # Checked here, in the main shell, rather than only inside
    # extract_rerun_ids(): that function is consumed via `< <(...)` process
    # substitution below, which runs in a subshell -- an exit there would only
    # end the subshell, not this script, and the wrong (generic) exit code
    # would propagate instead.
    echo "ERROR: no summary.txt found at $RERUN_SUMMARY_FILE" >&2
    exit 2
  fi
fi

[ -n "$OUTDIR" ] || OUTDIR="ciLogs_vlm_max_context_$(date +%Y%m%d_%H%M%S)"

# --list and --dry-run must not litter the tree with an empty log directory.
if [ "$LIST_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$OUTDIR" || exit 2
  OUTDIR="$(readlink -f "$OUTDIR")"
  RUN_LOG="$OUTDIR/run.log"
  SUMMARY="$OUTDIR/summary.txt"
else
  # $OUTDIR is a name only here, not a real directory (LIST_ONLY exits before
  # ever reaching the final summary block below; DRY_RUN's --group-by-engine/
  # --per-test path does not exit early, and its `continue`-only loop still
  # falls through to that same block, so SUMMARY has to be a real, already-
  # existing path here too -- not $OUTDIR/summary.txt, which mkdir never ran for).
  RUN_LOG="/dev/null"
  SUMMARY="/dev/null"
fi

# --- effective config --------------------------------------------------------
# suite_config.json is the only thing ctx_config.py reads; there are no more
# MAX_CTX_*/VLM_MAX_CTX_* env vars to export. --tier/--tp/--models/--batch instead
# patch a copy of it for this invocation, and every pytest process below is pointed
# at that copy via MAX_CTX_CONFIG_FILE (the one env var ctx_config.py still reads --
# a file locator, not a knob). A real run keeps its patched copy in $OUTDIR as a
# record of exactly what config that run used; --list/--dry-run use a scratch temp
# file since they never create $OUTDIR.
if [ "$LIST_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
  EFFECTIVE_CONFIG="$(mktemp /tmp/suite_config.XXXXXX.json)"
  trap 'rm -f "$EFFECTIVE_CONFIG"' EXIT
  SUMMARY_XLSX=""
else
  EFFECTIVE_CONFIG="$OUTDIR/suite_config.effective.json"
  SUMMARY_XLSX="$OUTDIR/vlm_max_context_summary.xlsx"
fi

python3 - "$EFFECTIVE_CONFIG" "$TIER_OVERRIDE" "$TP_OVERRIDE" "$MODELS_OVERRIDE" \
  "$BATCH_OVERRIDE" "$MODALITIES_OVERRIDE" "$PL_OVERRIDE" "$GL_OVERRIDE" \
  "$OUTDIR" "$LIST_ONLY" "$DRY_RUN" << 'PYEOF'
import json
import sys

out, tier, tp, models, batch, modalities, pl, gl, outdir, list_only, dry_run = sys.argv[1:]

with open("suite_config.json") as f:
    cfg = json.load(f)

if tier:
    cfg["sweep"]["tier"] = tier
if tp:
    cfg["selection"]["tp_sizes"] = [int(v) for v in tp.split(",") if v.strip()]
if models:
    cfg["selection"]["models"] = [v.strip() for v in models.split(",") if v.strip()]
if batch:
    cfg["selection"]["batch_sizes"] = [int(v) for v in batch.split(",") if v.strip()]
if modalities:
    cfg["selection"]["modalities"] = [
        v.strip() for v in modalities.split(",") if v.strip()
    ]
# --pl/--gl mean "run exactly this point," which overrides any --tier given
# alongside it (bash already validated both are present and integers).
if pl and gl:
    cfg["sweep"]["tier"] = "custom"
    cfg["sweep"]["custom_targets"] = [{"prompt_len": int(pl), "gen_len": int(gl)}]
# --list/--dry-run never create $OUTDIR, so leave the configured summary_file alone.
if list_only == "0" and dry_run == "0":
    cfg["reporting"]["summary_file"] = f"{outdir}/vlm_max_context_summary.xlsx"

with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF

export MAX_CTX_CONFIG_FILE="$EFFECTIVE_CONFIG"
DISPLAY_TIER="$(python3 -c "import json; print(json.load(open('$EFFECTIVE_CONFIG'))['sweep']['tier'])")"
DISPLAY_TP="$(python3 -c "import json; print(','.join(str(v) for v in json.load(open('$EFFECTIVE_CONFIG'))['selection']['tp_sizes']))")"

# Unbuffered so a killed test still leaves its output on disk.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

log() { echo "$@" | tee -a "$RUN_LOG"; }

# --- process cleanup --------------------------------------------------------

# Kill everything left in a pytest's process group. Safe by construction: the pgid
# was created for that pytest, so nothing else can be a member.
reap_group() {
  local pgid="$1" waited=0 remaining
  remaining="$(pgrep -g "$pgid" 2>/dev/null | wc -l)"
  [ "$remaining" -eq 0 ] && return 0

  log "  reaping $remaining process(es) left in group $pgid"
  kill -TERM -- "-$pgid" 2>/dev/null
  while [ "$waited" -lt "$GRACE" ]; do
    pgrep -g "$pgid" >/dev/null 2>&1 || { log "  group exited after TERM"; return 0; }
    sleep 1
    waited=$((waited + 1))
  done

  remaining="$(pgrep -g "$pgid" 2>/dev/null | wc -l)"
  log "  $remaining process(es) ignored TERM for ${GRACE}s -- sending KILL"
  kill -KILL -- "-$pgid" 2>/dev/null
  sleep 2
}

# Opt-in sweep for vLLM processes that left their group (double-forked, re-parented).
# Three guards keep this from touching unrelated work: the process must be ours, it
# must be younger than this script, and its *executable name* must be python -- not
# merely its arguments, or a shell whose command line happens to mention vllm (a grep,
# an editor, this script's own invocation) would match and be killed.
reap_strays() {
  [ "$REAP_STRAYS" -eq 1 ] || return 0
  local elapsed=$((SECONDS - SCRIPT_START + 5)) victims
  victims="$(ps -u "$(id -un)" -o pid=,etimes=,comm=,args= 2>/dev/null | awk -v max="$elapsed" -v self="$$" '
    $1 != self && $2 <= max && $3 ~ /^python/ && /(VLLM|vllm|EngineCore|VllmWorker|spawn_main)/ { print $1 }')"
  [ -z "$victims" ] && return 0

  log "  stray vLLM processes younger than this run: $(echo "$victims" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill -TERM $victims 2>/dev/null
  sleep 3
  # shellcheck disable=SC2086
  kill -KILL $victims 2>/dev/null
}

# Advisory only -- never fails the run, just records what the devices look like.
device_status() {
  [ -x "$QAIC_UTIL" ] || return 0
  local output ready
  output="$("$QAIC_UTIL" -q 2>/dev/null)" || {
    log "  device status unavailable ($QAIC_UTIL failed -- may need sudo)"
    return 0
  }
  ready="$(grep -c "Status:Ready" <<< "$output")"
  log "  devices reporting Status:Ready: $ready"
}

CURRENT_PGID=""
on_interrupt() {
  echo
  log "Interrupted -- cleaning up"
  [ -n "$CURRENT_PGID" ] && reap_group "$CURRENT_PGID"
  reap_strays
  exit 130
}
trap on_interrupt INT TERM

# --- collection -------------------------------------------------------------
# Collected ids are normalised to <file>::<test>[...] so they can be handed back to
# pytest from this directory whatever rootdir pytest picks.
collect_ids() {
  local file="$1"
  local -a kargs=()
  [ -n "$KFILTER" ] && kargs=(-k "$KFILTER")
  python -m pytest "$file" --collect-only -q -p no:cacheprovider "${kargs[@]}" 2>/dev/null \
    | sed -n "s|^.*\(${file}::.*\)$|\1|p"
}

# Cases that share an engine are the ones worth grouping. Engine identity is
# (model, TP, engine_len, batch_size), and engine_len follows from the
# (prompt_len, gen_len) point, so the key is model + tp + pl + gl + bs and only the
# modality varies within a batch. Sweep ids look like
# <Model>-tp<N>-<modality>-pl<PL>-gl<GL>-bs<N>; boundary ids look like <Model>-tp<N>
# and key on (model, tp) too -- each (model, tp) pair needs its own device count
# under a device-pool run (scheduler.py) even though sequentially they could all
# share one process (three tests per model, one shared ceiling-sized engine).
group_key_for() {
  local id="$1" params key
  case "$id" in
    test_ctx_sweep.py*)
      # Strip to the bracketed parametrisation, then drop the modality field: what is
      # left identifies the engine. Keying on the model matters now that several are
      # selected at once -- two models at the same point are two different engines.
      params="$(printf '%s' "$id" | sed -n 's/^.*\[\(.*\)\]$/\1/p')"
      key="$(printf '%s' "$params" | sed -n \
        's/^\(.*\)-tp\([0-9][0-9]*\)-[a-z_]*-pl\([0-9][0-9]*\)-gl\([0-9][0-9]*\)-bs\([0-9][0-9]*\)$/sweep-\1-tp\2-pl\3-gl\4-bs\5/p')"
      # An id that does not parse falls back to its own group rather than silently
      # joining every other unparsed id in one oversized batch.
      if [ -n "$key" ]; then echo "$key"; else echo "$id"; fi
      ;;
    test_ctx_boundary.py*)
      params="$(printf '%s' "$id" | sed -n 's/^.*\[\(.*\)\]$/\1/p')"
      key="$(printf '%s' "$params" | sed -n 's/^\(.*\)-tp\([0-9][0-9]*\)$/boundary-\1-tp\2/p')"
      if [ -n "$key" ]; then echo "$key"; else echo "$id"; fi
      ;;
    *)                  echo "${id%%::*}" ;;
  esac
}

# The device count a batch (or a single raw node id, in --per-test mode) needs --
# always its TP, which every id/key shape group_key_for() produces embeds as
# "-tp<N>" somewhere. Used only when --device-ids hands off to scheduler.py.
tp_for_batch() {
  printf '%s' "$1" | grep -oE '\-tp[0-9]+' | tail -1 | grep -oE '[0-9]+'
}

# --rerun-failed: pull FAIL/TIMEOUT/ERROR/NOT-RUN entries out of an earlier run's
# summary.txt (written identically by the sequential loop and by scheduler.py --
# see their matching "%-8s %6ss  %s" / "{:8s} {:6.0f}s  " formats) and print one
# real pytest node id per line.
#
# A --per-test entry's name is already a node id. A --group-by-engine entry's
# name is the synthetic key group_key_for() built, not a node id -- to expand it
# back to its real member node ids this looks up that name in the same
# directory's jobs.manifest (tab-separated index/name/num_devices/nodeids, the
# latter '|'-joined), which only exists for a --device-ids run. A
# --group-by-engine run without --device-ids has no other record of which node
# ids made up a batch, so that combination can't be expanded here.
extract_rerun_ids() {
  local target="$1" summary_file run_dir jobs_manifest name
  if [ -d "$target" ]; then
    run_dir="$target"
    summary_file="$target/summary.txt"
  else
    summary_file="$target"
    run_dir="$(dirname "$target")"
  fi
  if [ ! -f "$summary_file" ]; then
    # Defensive only: the main script already checked this (see
    # RERUN_SUMMARY_FILE above) before ever calling this function, since an
    # exit from here -- called via `< <(...)` process substitution -- would
    # only end that subshell, not the script.
    return 0
  fi
  jobs_manifest="$run_dir/jobs.manifest"

  local -a names=()
  while IFS= read -r name; do
    names+=("$name")
  done < <(grep -E '^(FAIL|TIMEOUT|ERROR|NOT RUN)[[:space:]]+[0-9]+s[[:space:]]+' "$summary_file" \
             | sed -E 's/^(FAIL|TIMEOUT|ERROR|NOT RUN)[[:space:]]+[0-9]+s[[:space:]]+//')

  if [ "${#names[@]}" -eq 0 ]; then
    echo "No FAIL/TIMEOUT/ERROR/NOT-RUN entries found in $summary_file" >&2
    exit 5
  fi

  for name in "${names[@]}"; do
    case "$name" in
      *.py::*)
        echo "$name"
        ;;
      *)
        if [ -f "$jobs_manifest" ]; then
          awk -F'\t' -v key="$name" '$2 == key { print $4 }' "$jobs_manifest" | tr '|' '\n'
        else
          echo "WARNING: '$name' looks like a --group-by-engine batch name, but no" >&2
          echo "  jobs.manifest was found next to $summary_file to expand it back to" >&2
          echo "  node ids -- skipping it. Re-run with --device-ids (which always writes" >&2
          echo "  jobs.manifest) or --per-test to make this expandable next time." >&2
        fi
        ;;
    esac
  done
}

log "=== VLM max-context run: $(date) ==="
log "dir=$SCRIPT_DIR mode=$MODE tier=$DISPLAY_TIER tp=$DISPLAY_TP config=$EFFECTIVE_CONFIG"
log "logs=$OUTDIR timeout=${TIMEOUT}s settle=${SETTLE}s reap_strays=$REAP_STRAYS"
[ ${#PYTEST_EXTRA[@]} -gt 0 ] && log "extra pytest args: ${PYTEST_EXTRA[*]}"

ALL_IDS=()
if [ -n "$RERUN_FAILED_OVERRIDE" ]; then
  declare -A _seen_ids=()
  while IFS= read -r id; do
    if [ -n "$id" ] && [ -z "${_seen_ids[$id]+set}" ]; then
      _seen_ids["$id"]=1
      ALL_IDS+=("$id")
    fi
  done < <(extract_rerun_ids "$RERUN_FAILED_OVERRIDE")
  log "re-running ${#ALL_IDS[@]} failed/timed-out/not-run test(s) from $RERUN_FAILED_OVERRIDE"
else
  for file in "${TEST_FILES[@]}"; do
    [ -f "$file" ] || { log "skipping missing $file"; continue; }
    while IFS= read -r id; do
      [ -n "$id" ] && ALL_IDS+=("$id")
    done < <(collect_ids "$file")
  done
fi

if [ ${#ALL_IDS[@]} -eq 0 ]; then
  log "No tests collected. Check -k / suite_config.json (or --tier/--tp/--models/--batch), or run pytest --collect-only by hand."
  exit 5
fi
log "collected ${#ALL_IDS[@]} test(s)"

# --- build the batches ------------------------------------------------------
# BATCH_IDS[i] holds a newline-separated list of node ids to run in one pytest.
BATCH_NAMES=()
BATCH_IDS=()
if [ "$MODE" = "per-test" ]; then
  for id in "${ALL_IDS[@]}"; do
    BATCH_NAMES+=("$id")
    BATCH_IDS+=("$id")
  done
else
  declare -A GROUPED=()
  for id in "${ALL_IDS[@]}"; do
    key="$(group_key_for "$id")"
    if [ -z "${GROUPED[$key]+set}" ]; then
      GROUPED["$key"]="$id"
      BATCH_NAMES+=("$key")
    else
      GROUPED["$key"]="${GROUPED[$key]}"$'\n'"$id"
    fi
  done
  for key in "${BATCH_NAMES[@]}"; do
    BATCH_IDS+=("${GROUPED[$key]}")
  done
fi

if [ "$LIST_ONLY" -eq 1 ]; then
  for index in "${!BATCH_NAMES[@]}"; do
    echo "--- batch $((index + 1)): ${BATCH_NAMES[$index]}"
    printf '      %s\n' ${BATCH_IDS[$index]//$'\n'/ }
  done
  exit 0
fi

# --- device-pool dispatch ----------------------------------------------------
# --device-ids hands the whole run off to scheduler.py, which runs batches
# concurrently across a device pool instead of this script's own one-at-a-time
# loop below. Every batch already has a well-defined device count (its TP, via
# tp_for_batch) whether MODE is per-test or group-by-engine.
if [ -n "$DEVICE_IDS_OVERRIDE" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    for index in "${!BATCH_NAMES[@]}"; do
      name="${BATCH_NAMES[$index]}"
      num_devices="$(tp_for_batch "$name")"
      [ -z "$num_devices" ] && num_devices=0
      log "  would dispatch [$((index + 1))/${#BATCH_NAMES[@]}] $name (needs $num_devices device(s))"
    done
    exit 0
  fi

  MANIFEST="$OUTDIR/jobs.manifest"
  : > "$MANIFEST"
  for index in "${!BATCH_NAMES[@]}"; do
    name="${BATCH_NAMES[$index]}"
    num_devices="$(tp_for_batch "$name")"
    # No -tp<N> in the id at all (test_token_math_selftest.py: parametrized only
    # by model, no device needed) -- run it immediately as a 0-device job rather
    # than dropping it. A real sweep/boundary id always has -tp<N>, so this only
    # ever fires for genuinely device-free tests, never a parse failure on one
    # that actually needs a device.
    [ -z "$num_devices" ] && num_devices=0
    ids_joined="$(printf '%s' "${BATCH_IDS[$index]}" | tr '\n' '|')"
    printf '%s\t%s\t%s\t%s\n' "$index" "$name" "$num_devices" "$ids_joined" >> "$MANIFEST"
  done

  log "Dispatching ${#BATCH_NAMES[@]} batch(es) across device pool: $DEVICE_IDS_OVERRIDE"
  STOP_FLAG=()
  [ "$STOP_ON_FAIL" -eq 1 ] && STOP_FLAG=(--stop-on-fail)
  python3 "$SCRIPT_DIR/scheduler.py" "$MANIFEST" \
    --device-ids "$DEVICE_IDS_OVERRIDE" \
    --output-dir "$OUTDIR" \
    --summary-file "$SUMMARY" \
    --timeout "$TIMEOUT" \
    --grace "$GRACE" \
    --cooldown "$SETTLE" \
    "${STOP_FLAG[@]}" \
    -- "${PYTEST_EXTRA[@]}" 2>&1 | tee -a "$RUN_LOG"
  SCHED_RC=${PIPESTATUS[0]}
  exit "$SCHED_RC"
fi

# --- run --------------------------------------------------------------------
PASSED=0; FAILED=0; SKIPPED=0; TIMEDOUT=0
declare -a RESULT_LINES=()
TOTAL=${#BATCH_NAMES[@]}

classify() {
  local rc="$1" logfile="$2" tail_lines
  case "$rc" in
    124|137) echo "TIMEOUT"; return ;;
    5)       echo "NOTESTS"; return ;;
  esac
  tail_lines="$(tail -5 "$logfile" 2>/dev/null)"
  if [ "$rc" -eq 0 ]; then
    if grep -qE "[0-9]+ skipped" <<< "$tail_lines" && ! grep -qE "[0-9]+ passed" <<< "$tail_lines"; then
      echo "SKIP"
    else
      echo "PASS"
    fi
  else
    echo "FAIL"
  fi
}

for index in "${!BATCH_NAMES[@]}"; do
  name="${BATCH_NAMES[$index]}"
  mapfile -t ids <<< "${BATCH_IDS[$index]}"
  slug="$(printf '%s' "$name" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-120)"
  logfile="$OUTDIR/$(printf '%03d' $((index + 1)))_${slug}.log"

  log ""
  log "[$((index + 1))/$TOTAL] $name"
  log "  -> $logfile"

  if [ "$DRY_RUN" -eq 1 ]; then
    log "  would run: python -m pytest -s ${ids[*]} ${PYTEST_EXTRA[*]}"
    continue
  fi

  started=$SECONDS
  {
    echo "### $(date) :: $name"
    echo "### pytest -s ${ids[*]} ${PYTEST_EXTRA[*]}"
    echo
  } > "$logfile"

  # Job control puts this pytest in its own process group (pgid == its pid), which is
  # what makes reap_group safe: the group contains this pytest and its children only.
  set -m
  timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
    python -m pytest -s -p no:cacheprovider "${ids[@]}" "${PYTEST_EXTRA[@]}" \
    >> "$logfile" 2>&1 &
  pgid=$!
  set +m
  CURRENT_PGID="$pgid"

  wait "$pgid"
  rc=$?
  CURRENT_PGID=""
  elapsed=$((SECONDS - started))

  status="$(classify "$rc" "$logfile")"
  case "$status" in
    PASS)    PASSED=$((PASSED + 1)) ;;
    SKIP)    SKIPPED=$((SKIPPED + 1)) ;;
    TIMEOUT) TIMEDOUT=$((TIMEDOUT + 1)) ;;
    *)       FAILED=$((FAILED + 1)) ;;
  esac
  log "  $status (rc=$rc) in ${elapsed}s"
  RESULT_LINES+=("$(printf '%-8s %6ss  %s' "$status" "$elapsed" "$name")")

  # Always clean up, pass or fail: a leaked worker breaks the *next* test.
  reap_group "$pgid"
  reap_strays
  device_status

  if [ "$status" = "FAIL" ] || [ "$status" = "TIMEOUT" ]; then
    log "  last lines of $logfile:"
    tail -15 "$logfile" | sed 's/^/    /' | tee -a "$RUN_LOG"
    if [ "$STOP_ON_FAIL" -eq 1 ]; then
      log "--stop-on-fail: stopping after $name"
      break
    fi
  fi

  # Let the devices settle before the next engine claims them.
  if [ "$index" -lt $((TOTAL - 1)) ] && [ "$SETTLE" -gt 0 ]; then
    log "  settling ${SETTLE}s"
    sleep "$SETTLE"
  fi
done

# --- summary ----------------------------------------------------------------
{
  echo "=== VLM max-context summary: $(date) ==="
  echo "mode=$MODE tier=$DISPLAY_TIER tp=$DISPLAY_TP"
  echo
  printf '%s\n' "${RESULT_LINES[@]}"
  echo
  echo "passed=$PASSED failed=$FAILED skipped=$SKIPPED timeout=$TIMEDOUT of $TOTAL"
  echo "logs: $OUTDIR"
  [ -f "$SUMMARY_XLSX" ] && echo "perf tables: $SUMMARY_XLSX"
} | tee "$SUMMARY" | tee -a "$RUN_LOG"

[ $((FAILED + TIMEDOUT)) -eq 0 ] || exit 1
exit 0
