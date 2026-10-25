"""Summarise a `host.sh` run as Markdown.

Usage: python3 summarise.py <run dir> <reference sha> [<sha>...]

Every comparison is against the first commit.

- perf: per-operation counters, `(run(M) - run(0)) / M`.
- Criterion: for each benchmark and round, the ratio of a commit's median to the
  reference's median in the same round. Reports the median ratio with a
  bootstrap 95% confidence interval. Pairing within rounds cancels slow drift.
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics
import sys

COUNTERS = ("instructions", "cycles", "branch-misses", "L1-dcache-load-misses")
SHORT = {
    "instructions": "instr",
    "cycles": "cycles",
    "branch-misses": "br-miss",
    "L1-dcache-load-misses": "L1-miss",
}


# MARK: perf


def read_perf(path: pathlib.Path) -> dict[str, float]:
    """Read a `perf stat -x,` CSV into {event: count}."""
    counts = {}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[0].replace(".", "").isdigit():
            counts[fields[2].split(":")[0]] = float(fields[0])
    return counts


def perf_section(run: pathlib.Path, shas: list[str]) -> list[str]:
    """Per-operation counters for each workload and commit."""
    perf = run / "perf"
    if not perf.is_dir():
        return []
    # {(workload, n): {sha: {event: per-op}}}
    table: dict[tuple[str, int], dict[str, dict[str, float]]] = collections.defaultdict(
        dict
    )
    for zero in perf.glob("*.0.csv"):
        sha, workload, n, _, _ = zero.name.split(".")
        (full,) = [p for p in perf.glob(f"{sha}.{workload}.{n}.*.csv") if p != zero]
        iters = int(full.name.split(".")[3])
        a, z = read_perf(full), read_perf(zero)
        table[(workload.replace("_", "/", 1), int(n))][sha] = {
            k: (a[k] - z.get(k, 0.0)) / iters for k in a
        }

    out = ["## perf stat (per operation)", ""]
    header = "| benchmark | commit | " + " | ".join(SHORT[c] for c in COUNTERS) + " |"
    out += [header, "|" + "---|" * (len(COUNTERS) + 2)]
    for key in sorted(table):
        ref = table[key].get(shas[0])
        for sha in shas:
            row = table[key].get(sha)
            if row is None:
                continue
            cells = []
            for c in COUNTERS:
                v = row.get(c, float("nan"))
                cell = f"{v:.1f}"
                if ref and sha != shas[0] and ref.get(c):
                    cell += f" ({v / ref[c] - 1:+.2%})"
                cells.append(cell)
            out.append(f"| {key[0]}/{key[1]} | {sha} | " + " | ".join(cells) + " |")
    return [*out, ""]


# MARK: Criterion


def criterion_section(run: pathlib.Path, shas: list[str]) -> list[str]:
    """Paired per-round ratios against the reference commit."""
    root = run / "criterion"
    if not root.is_dir():
        return []
    # {(benchmark, round): {sha: median ns}}
    data: dict[tuple[str, int], dict[str, float]] = collections.defaultdict(dict)
    for est in root.rglob("estimates.json"):
        baseline = est.parent.name
        if not baseline.startswith("r") or "-" not in baseline:
            continue
        rnd, sha = baseline[1:].split("-", 1)
        if not rnd.isdigit():
            continue
        bid = json.loads((est.parent / "benchmark.json").read_text())["full_id"]
        median = json.loads(est.read_text())["median"]["point_estimate"]
        data[(bid, int(rnd))][sha] = median

    rng = random.Random(1)
    ref = shas[0]
    out = [
        "## Criterion (paired median ratio vs reference, bootstrap 95% CI)",
        "",
        f"| benchmark | {ref} ns | "
        + " | ".join(f"{s} change | CI" for s in shas[1:])
        + " |",
        "|" + "---|" * (2 + 2 * len(shas[1:])),
    ]
    for bid in sorted({b for b, _ in data}):
        rounds = sorted(r for b, r in data if b == bid)
        ref_ns = [data[(bid, r)][ref] for r in rounds if ref in data[(bid, r)]]
        cells = [f"{statistics.median(ref_ns):.1f}" if ref_ns else "-"]
        for sha in shas[1:]:
            ratios = [
                data[(bid, r)][sha] / data[(bid, r)][ref] - 1
                for r in rounds
                if sha in data[(bid, r)] and ref in data[(bid, r)]
            ]
            if not ratios:
                cells += ["-", "-"]
                continue
            boots = sorted(
                statistics.median(rng.choices(ratios, k=len(ratios)))
                for _ in range(5000)
            )
            lo, hi = boots[125], boots[4874]
            cells += [f"{statistics.median(ratios):+.2%}", f"[{lo:+.2%}, {hi:+.2%}]"]
        out.append(f"| {bid} | " + " | ".join(cells) + " |")
    return [*out, "", f"Rounds: {max((r for _, r in data), default=0)}", ""]


def main() -> None:
    """Print the summary for the run directory and commits on the command line."""
    run = pathlib.Path(sys.argv[1])
    shas = sys.argv[2:]
    lines = [f"# Benchmark run `{run.name}`", "", f"Reference: `{shas[0]}`", ""]
    env = run / "env.txt"
    if env.exists():
        lines += ["```text", env.read_text().strip(), "```", ""]
    lines += perf_section(run, shas)
    lines += criterion_section(run, shas)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
