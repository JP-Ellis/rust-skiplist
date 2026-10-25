#!/usr/bin/env bash
# Dispatches a `hot` benchmark run to the benchmark host and collects the result.
#
# Usage:
#   scripts/bench/remote.sh run [options] <ref> [<ref>...]
#   scripts/bench/remote.sh status <label>
#   scripts/bench/remote.sh wait <label>
#   scripts/bench/remote.sh fetch <label>
#   scripts/bench/remote.sh stop <label>
#
# `run` compares the refs against the first one. It ships the commits as a git
# bundle (they need not be pushed), and ships `benches/hot.rs` from the working
# tree, so every commit is measured with the same harness. The job runs as a
# transient systemd user unit, `skiplist-hot-<label>`, and survives SSH drops.
#
# Run options:
#   -l <label>   run name (default: date and time)
#   -m <mode>    perf, criterion or all (default all)
#   -r <rounds>  Criterion rounds (default 10)
#   -f <regex>   Criterion filter, e.g. '^set/' (default: everything)
#   -w           wait for completion, then fetch the summary
#
# Environment:
#   BENCH_HOST   host to run on (required)
#   QUIET_UNITS  user units `host.sh` stops for the run (see `host.sh`)
#   WAIT_UNITS   user units whose jobs `host.sh` waits out (see `host.sh`)
#
# Results land in target/bench-remote/<label>/ (summary.md, progress.log).
# The host must have the performance governor set and boost disabled; that
# needs root, so `host.sh` checks and refuses rather than changing it.

set -euo pipefail

HOST=${BENCH_HOST:?set BENCH_HOST to the benchmark host}
CLONE=/tmp/skiplist-bench
REMOTE_ROOT=$CLONE/runs
REPO_URL=https://github.com/JP-Ellis/rust-skiplist.git
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)

remote() { ssh -T -o BatchMode=yes "$HOST" "$@"; }

usage() {
  sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" >&2
  exit 2
}

# MARK: Commands

cmd_run() {
  local label mode=all rounds=10 filter="" wait=0 opt
  label=$(date '+%Y%m%d-%H%M%S')
  while getopts 'l:m:r:f:w' opt; do
    case $opt in
    l) label=$OPTARG ;;
    m) mode=$OPTARG ;;
    r) rounds=$OPTARG ;;
    f) filter=$OPTARG ;;
    w) wait=1 ;;
    *) usage ;;
    esac
  done
  shift $((OPTIND - 1))
  (($# > 0)) || usage

  local -a shas refs
  local ref sha
  for ref in "$@"; do
    sha=$(git -C "$ROOT" rev-parse --short=12 "$ref^{commit}")
    shas+=("$sha")
    refs+=("refs/bench/$sha")
    git -C "$ROOT" update-ref "refs/bench/$sha" "$sha"
  done

  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  git -C "$ROOT" bundle create -q "$tmp/commits.bundle" "${refs[@]}"
  for ref in "${refs[@]}"; do git -C "$ROOT" update-ref -d "$ref"; done
  cp "$ROOT/benches/hot.rs" "$HERE/host.sh" "$HERE/summarise.py" "$tmp/"

  local run=$REMOTE_ROOT/$label unit=skiplist-hot-$label
  if remote "systemctl --user list-units --state=active --no-legend 'skiplist-hot-*'" | grep -q .; then
    echo "another skiplist-hot run is active on $HOST:" >&2
    remote "systemctl --user list-units --state=active --no-legend 'skiplist-hot-*'" >&2
    exit 1
  fi

  remote "mkdir -p '$run' && { [ -d '$CLONE/.git' ] || git clone -q '$REPO_URL' '$CLONE'; }"
  scp -q "$tmp"/* "$HOST:$run/"
  remote "cd '$CLONE' && git fetch -q '$run/commits.bundle' 'refs/bench/*:refs/bench/*'"

  remote "systemd-run --user --quiet --collect --unit='$unit' \
    --setenv=MODE='$mode' --setenv=ROUNDS='$rounds' --setenv=FILTER='$filter' \
    --setenv=QUIET_UNITS='${QUIET_UNITS:-}' --setenv=WAIT_UNITS='${WAIT_UNITS:-}' \
    --setenv=PATH=\"\$PATH\" \
    -p ExecStopPost='/usr/bin/env bash $run/host.sh restore $run' \
    bash '$run/host.sh' run '$run' ${shas[*]}"

  echo "started $unit on $HOST: ${shas[*]} (mode $mode, rounds $rounds)"
  echo "follow:  scripts/bench/remote.sh status $label"
  if ((wait)); then
    cmd_wait "$label"
  fi
}

cmd_status() {
  local label=${1:?missing label}
  remote "systemctl --user is-active skiplist-hot-$label || true; tail -n 5 '$REMOTE_ROOT/$label/progress.log'"
}

cmd_wait() {
  local label=${1:?missing label}
  remote "while systemctl --user is-active --quiet skiplist-hot-$label; do sleep 30; done"
  cmd_fetch "$label"
}

cmd_fetch() {
  local label=${1:?missing label} dest
  dest=$ROOT/target/bench-remote/$label
  mkdir -p "$dest"
  scp -q "$HOST:$REMOTE_ROOT/$label/progress.log" "$HOST:$REMOTE_ROOT/$label/env.txt" "$dest/"
  if scp -q "$HOST:$REMOTE_ROOT/$label/summary.md" "$dest/" 2>/dev/null; then
    cat "$dest/summary.md"
  else
    echo "no summary yet; last progress:" >&2
    tail -n 5 "$dest/progress.log" >&2
  fi
}

cmd_stop() {
  local label=${1:?missing label}
  remote "systemctl --user stop skiplist-hot-$label"
}

# MARK: Main

command=${1:-}
(($# > 0)) && shift
case $command in
run) cmd_run "$@" ;;
status) cmd_status "$@" ;;
wait) cmd_wait "$@" ;;
fetch) cmd_fetch "$@" ;;
stop) cmd_stop "$@" ;;
*) usage ;;
esac
