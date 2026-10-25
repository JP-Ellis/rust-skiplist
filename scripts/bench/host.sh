#!/usr/bin/env bash
# Runs the `hot` benchmark suite for a set of commits on the benchmark host.
#
# Usage:
#   host.sh run <run dir> <sha>...
#   host.sh restore <run dir>
#
# `run` builds one `hot` binary per commit, quiets the machine, then measures:
#
# - perf: `perf stat` counters per operation (deterministic instruction counts).
# - criterion: shuffled, interleaved Criterion rounds compared offline.
#
# The run directory must contain `hot.rs` (the harness) and `summarise.py`.
# Every commit is built with that same harness, so all commits run identical
# workloads, including commits that predate the harness.
#
# `restore` restarts the services `run` stopped. `run` calls it on exit, and
# `remote.sh` also installs it as the unit's ExecStopPost, so the services come
# back even when the run is killed.
#
# Environment:
#   MODE        perf, criterion or all (default all)
#   ROUNDS      Criterion rounds (default 10)
#   FILTER      Criterion filter regex (default: every benchmark)
#   CRIT_ARGS   extra Criterion arguments
#   PERF_SCALE  divides the perf step counts that `hot list` suggests (default 1)
#   CPU         CPU to pin to (default 5; its SMT sibling should stay idle)
#   QUIET_UNITS space-separated user units to stop for the run and restore after,
#               such as backup timers and file-sync services
#   WAIT_UNITS  space-separated user units whose running jobs the run waits out
#   ALLOW_NOISY set to 1 to run without the performance governor or with boost

set -euo pipefail

CLONE=${CLONE:-/tmp/skiplist-bench}
MODE=${MODE:-all}
ROUNDS=${ROUNDS:-10}
FILTER=${FILTER:-}
CRIT_ARGS=${CRIT_ARGS:-}
PERF_SCALE=${PERF_SCALE:-1}
CPU=${CPU:-5}
EVENTS=instructions:u,cycles:u,branches:u,branch-misses:u,L1-dcache-load-misses:u

# Only the units found active are stopped and restored.
read -ra QUIET <<<"${QUIET_UNITS:-}"
read -ra WAIT <<<"${WAIT_UNITS:-}"

# MARK: Helpers

log() { echo "$(date '+%F %T') $*" | tee -a "$RUN/progress.log"; }

restore() {
  local state=$RUN/stopped-units
  [[ -f $state ]] || return 0
  local failed=$state.failed
  : >"$failed"
  while read -r unit; do
    [[ -n $unit ]] || continue
    if systemctl --user start "$unit"; then
      log "restarted $unit"
    else
      log "FAILED to restart $unit; kept in stopped-units for retry"
      echo "$unit" >>"$failed"
    fi
  done <"$state"
  if [[ -s $failed ]]; then
    mv "$failed" "$state"
  else
    rm -f "$failed" "$state"
  fi
}

quiet_down() {
  local unit
  : >"$RUN/stopped-units"
  for unit in "${QUIET[@]}"; do
    if systemctl --user is-active --quiet "$unit"; then
      echo "$unit" >>"$RUN/stopped-units"
      systemctl --user stop "$unit"
      log "stopped $unit"
    fi
  done
  for unit in "${WAIT[@]}"; do
    while systemctl --user is-active --quiet "$unit"; do
      log "waiting for $unit to finish"
      sleep 30
    done
  done
}

check_cpu() {
  local governor boost
  governor=$(cat "/sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor")
  boost=$(cat /sys/devices/system/cpu/cpufreq/boost)
  log "cpu $CPU governor $governor boost $boost"
  if [[ $governor != performance || $boost != 0 ]] && [[ ${ALLOW_NOISY:-0} != 1 ]]; then
    log "refusing to run: needs the performance governor and boost off (or ALLOW_NOISY=1)"
    exit 1
  fi
}

# Prints "<busy CPUs> <busy fraction of the pinned CPU>" over one second.
# Unlike the load average, this ignores the benchmark that has just finished.
busy() {
  local a b
  a=$(grep -E "^cpu |^cpu$CPU " /proc/stat)
  sleep 1
  b=$(grep -E "^cpu |^cpu$CPU " /proc/stat)
  paste <(echo "$a") <(echo "$b") | awk -v ncpu="$(nproc)" '
    {
      n = (NF - 2) / 2
      t0 = 0; t1 = 0
      for (i = 2; i <= n + 1; i++) { t0 += $i; t1 += $(i + n + 1) }
      i0 = $5 + $6; i1 = $(5 + n + 1) + $(6 + n + 1)
      frac = 1 - (i1 - i0) / (t1 - t0)
      if ($1 == "cpu") all = frac * ncpu; else pin = frac
    }
    END { printf "%.2f %.2f\n", all, pin }'
}

wait_quiet() {
  local all pin
  while :; do
    read -r all pin < <(busy)
    awk -v a="$all" -v p="$pin" 'BEGIN { exit !(a < 0.5 && p < 0.05) }' && break
    log "busy: $all CPUs, cpu $CPU at $pin; waiting" >&2
    sleep 20
  done
  echo "$all"
}

# MARK: Build

build() {
  local sha=$1 exe
  local key
  key=$sha-$(sha256sum "$RUN/hot.rs" | cut -c1-8)
  BIN[$sha]=$CLONE/bin/$key
  [[ -x ${BIN[$sha]} ]] && return 0
  mkdir -p "$CLONE/bin"
  git -C "$CLONE" checkout -q --force "$sha"
  cp "$RUN/hot.rs" "$CLONE/benches/hot.rs"
  if ! grep -q '^name = "hot"$' "$CLONE/Cargo.toml"; then
    printf '\n[[bench]]\nharness = false\nname = "hot"\nrequired-features = ["std"]\n' >>"$CLONE/Cargo.toml"
  fi
  log "building $sha"
  exe=$(cd "$CLONE" && cargo bench --bench hot --no-run 2>&1 | tee -a "$RUN/build.log" |
    sed -n 's/.*Executable .*(\(.*\))$/\1/p') || exe=
  [[ -n $exe ]] || {
    log "build of $sha failed; see build.log"
    exit 1
  }
  cp "$CLONE/$exe" "${BIN[$sha]}"
  git -C "$CLONE" checkout -q --force "$sha"
  git -C "$CLONE" clean -fdq benches
}

# MARK: Measure

run_perf() {
  local workload n sha iters
  mkdir -p "$RUN/perf"
  log "perf start"
  while read -r workload n iters; do
    iters=$((iters / PERF_SCALE))
    for sha in "${SHAS[@]}"; do
      local base=$RUN/perf/${sha}.${workload//\//_}.$n
      taskset -c "$CPU" perf stat -x, -e "$EVENTS" -o "$base.0.csv" \
        "${BIN[$sha]}" opcount "$workload" "$n" 0
      taskset -c "$CPU" perf stat -x, -r 3 -e "$EVENTS" -o "$base.$iters.csv" \
        "${BIN[$sha]}" opcount "$workload" "$n" "$iters"
    done
  done < <("${BIN[${SHAS[0]}]}" list)
  log "perf done"
}

run_criterion() {
  local round sha all
  local -a args
  read -r -a args <<<"$CRIT_ARGS"
  [[ -n $FILTER ]] && args+=("$FILTER")
  mkdir -p "$RUN/criterion-logs"
  for round in $(seq 1 "$ROUNDS"); do
    for sha in $(printf '%s\n' "${SHAS[@]}" | shuf); do
      all=$(wait_quiet)
      log "round $round $sha busy $all"
      CRITERION_HOME=$RUN/criterion taskset -c "$CPU" "${BIN[$sha]}" --bench --noplot \
        --save-baseline "r$round-$sha" "${args[@]}" >"$RUN/criterion-logs/r$round-$sha.log" 2>&1
    done
  done
  log "criterion done"
}

# MARK: Main

cmd=${1:?usage: host.sh run|restore <run dir> [sha...]}
RUN=${2:?missing run dir}
shift 2

case $cmd in
restore)
  restore
  exit 0
  ;;
run) ;;
*)
  echo "unknown command $cmd" >&2
  exit 2
  ;;
esac

SHAS=("$@")
((${#SHAS[@]} > 0)) || {
  echo "no commits given" >&2
  exit 2
}
declare -A BIN

log "run $(basename "$RUN"): mode $MODE, commits ${SHAS[*]}"
{
  uname -r
  grep -m1 'model name' /proc/cpuinfo
  rustc --version
  echo "MODE=$MODE ROUNDS=$ROUNDS FILTER=$FILTER CRIT_ARGS=$CRIT_ARGS PERF_SCALE=$PERF_SCALE CPU=$CPU"
} >"$RUN/env.txt"
check_cpu
for sha in "${SHAS[@]}"; do build "$sha"; done

trap restore EXIT
trap 'exit 143' TERM INT HUP
quiet_down

case $MODE in
perf) run_perf ;;
criterion) run_criterion ;;
all)
  run_perf
  run_criterion
  ;;
*)
  log "unknown MODE $MODE"
  exit 2
  ;;
esac

python3 "$RUN/summarise.py" "$RUN" "${SHAS[@]}" >"$RUN/summary.md"
log "all done"
