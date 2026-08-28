#!/usr/bin/env python3
"""
Nsight Compute (ncu) driver for the CUDA optimization benchmarks.

Profiles each optimization's before/after kernel pair and writes three
artifacts per optimization to results/:

    <slug>_<before|after>_<run-id>.ncu-rep   raw ncu report
    <slug>_check_<run-id>.txt               did each counter move as predicted
    <slug>_prompt_<run-id>.txt              full context for an AI analysis

<run-id> is one timestamp shared by every optimization in a run.

The check file only records whether expectations were met; it deliberately does
not speculate about causes.  The prompt file carries the complete source of the
optimization plus the measured numbers, so the analysis can be done against the
actual code.

Usage:
    # profile all twelve optimizations
    python run_ncu.py

    # profile a subset, by id and/or name substring
    python run_ncu.py --only 4,8
    python run_ncu.py --only "memory coalescing,bank"

Requirements:
    - ncu >= 2021.1 on PATH  (typically /usr/local/cuda/bin/ncu)
    - binaries built:  make build

Each binary runs a single variant per invocation ("before" or "after"), so the
profiled launch is always launch index 1: one warmup launch, then the timed
loop.  That is why --launch-skip is a constant here.
"""

import argparse
import csv
import datetime
import io
import os
import shutil
import subprocess
import sys
from typing import Optional

import optimizations
from optimizations import OPTIMIZATIONS, Metric, Optimization

RESULTS_DIR = os.path.join(optimizations.ROOT, "results")

# timeKernelMs issues exactly one warmup launch before the timed loop.
LAUNCH_SKIP = 1


def supported_metrics() -> set:
    """Counter names this device exposes.  Unsupported names are silently
    dropped by ncu, which would otherwise look like a flat result."""
    try:
        out = subprocess.run(["ncu", "--query-metrics"], capture_output=True,
                             text=True, timeout=120).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return set()
    return {line.split()[0] for line in out.splitlines()
            if line and not line[0].isspace()}


def base_counter(name: str) -> str:
    """'lts__t_sectors_op_read.sum' -> 'lts__t_sectors_op_read'."""
    return name.split(".", 1)[0]


def run_ncu(binary: str, variant: str, kernel: str, metrics: list[str],
            export_path: str) -> Optional[dict]:
    """Profile one kernel launch and return {counter: (value, unit)}.

    ncu suppresses its CSV output whenever --export is given, so the report is
    written first and then read back with --import.
    """
    profile = [
        "ncu",
        "--kernel-name",  kernel,
        "--launch-skip",  str(LAUNCH_SKIP),
        "--launch-count", "1",
        "--metrics",      ",".join(metrics),
        "--export",       export_path,
        "--force-overwrite",
        binary, variant,
    ]
    try:
        result = subprocess.run(profile, capture_output=True, text=True,
                                timeout=900)
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] ncu exceeded 15 min on {binary} {variant}",
              file=sys.stderr)
        return None
    except FileNotFoundError:
        print("ERROR: 'ncu' not found on PATH.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0 and result.stderr.strip():
        print(f"\n  [ncu stderr] {result.stderr.strip()}", file=sys.stderr)
        return None

    dump = subprocess.run(["ncu", "--import", export_path, "--csv",
                           "--page", "raw"],
                          capture_output=True, text=True, timeout=300)
    rows = list(csv.reader(io.StringIO(dump.stdout)))
    if len(rows) < 3:
        return None

    header, units, data = rows[0], rows[1], rows[2:]
    measured: dict[str, tuple[float, str]] = {}
    for name in metrics:
        if name not in header:
            continue
        col = header.index(name)
        values = []
        for row in data:
            try:
                # ncu writes thousands separators, e.g. "32,768".
                values.append(float(row[col].replace(",", "")))
            except (ValueError, IndexError):
                pass
        if values:
            measured[name] = (sum(values) / len(values), units[col])
    return measured


def fmt_val(measured: Optional[dict], counter: str) -> str:
    """Format one counter value, or 'n/a' if it was not collected."""
    if not measured or counter not in measured:
        return "n/a"
    value, _ = measured[counter]
    if abs(value) >= 1e9:
        return f"{value:.3e}"
    if abs(value) >= 1e4:
        return f"{value:.0f}"
    return f"{value:.2f}"


def unit_of(measured: Optional[dict], counter: str) -> str:
    if not measured or counter not in measured:
        return ""
    return measured[counter][1]


def pct_change(before: str, after: str) -> Optional[float]:
    try:
        b, a = float(before), float(after)
    except (ValueError, TypeError):
        return None
    return None if b == 0 else (a - b) / abs(b) * 100


def fmt_change(pct: Optional[float]) -> str:
    return "n/a" if pct is None else f"{pct:+.1f}%"


def verdict(pct: Optional[float], expected: str) -> str:
    """Compare observed direction against the prediction.  No interpretation."""
    if pct is None:
        return "NO DATA"
    moved = "flat" if abs(pct) < 1.0 else ("up" if pct > 0 else "down")
    if expected == "any":
        return f"INFO ({moved})"
    if moved == "flat":
        return "NOT MET (flat)"
    return "MET" if moved == expected else f"NOT MET ({moved})"


def collect_rows(opt: Optimization,
                 mb: Optional[dict], ma: Optional[dict]) -> list[dict]:
    """One row per metric: values, change, expectation, verdict."""
    rows = []
    for m in opt.metrics:
        before = fmt_val(mb, m.name)
        after  = fmt_val(ma, m.name)
        pct    = pct_change(before, after)
        unit   = unit_of(mb, m.name) or unit_of(ma, m.name)
        rows.append({
            "counter": m.name, "label": m.label, "why": m.why, "unit": unit,
            "before": before, "after": after, "pct": pct,
            "change": fmt_change(pct), "expected": m.expect,
            "verdict": verdict(pct, m.expect),
        })
    return rows


def metric_table(rows: list[dict], width: int) -> list[str]:
    out = [
        f"{'Metric':<16}{'Unit':<8}{'Before':>14}{'After':>14}{'Change':>10}"
        f"  {'Expected':<9}Result",
        "-" * width,
    ]
    for r in rows:
        out.append(f"{r['label']:<16}{r['unit']:<8}{r['before']:>14}"
                   f"{r['after']:>14}{r['change']:>10}  "
                   f"{r['expected']:<9}{r['verdict']}")
    return out


def build_check(opt: Optimization, run_id: str,
                rows: list[dict], reports: dict[str, str]) -> str:
    """Record which expectations the counters met.  Interpretation is the
    prompt's job, so nothing here speculates about causes."""
    W = 88
    out = [
        "=" * W,
        f"{opt.display}   (run {run_id})",
        f"  source  : {os.path.relpath(opt.source_path, optimizations.ROOT)}",
        f"  before  : {opt.before_kernel}",
        f"  after   : {opt.after_kernel}",
        f"  reports : {os.path.basename(reports['before'])}",
        f"            {os.path.basename(reports['after'])}",
        "=" * W,
        "",
    ]
    out += metric_table(rows, W)

    judged  = [r for r in rows if r["expected"] != "any" and r["pct"] is not None]
    met     = [r for r in judged if r["verdict"] == "MET"]
    not_met = [r for r in judged if r["verdict"].startswith("NOT MET")]
    no_data = [r for r in rows if r["pct"] is None]

    duration = next((r for r in rows if r["counter"] == "gpu__time_duration.sum"),
                    None)

    out += ["-" * W, "", "RESULT", "-" * W]
    if duration and duration["pct"] is not None:
        try:
            speedup = float(duration["before"]) / float(duration["after"])
            unit = duration["unit"] or ""
            out.append(f"Kernel time : {duration['before']} {unit} -> "
                       f"{duration['after']} {unit}  ({duration['change']}, "
                       f"{speedup:.2f}x)")
        except (ValueError, ZeroDivisionError):
            out.append(f"Kernel time : {duration['change']}")
    else:
        out.append("Kernel time : no data")

    out.append(f"Expectations: {len(met)}/{len(judged)} met")
    if met:
        out.append("  met      : " + ", ".join(f"{r['label']} {r['change']}"
                                               for r in met))
    if not_met:
        out.append("  not met  : " + ", ".join(
            f"{r['label']} {r['change']} (expected {r['expected']})"
            for r in not_met))
    if no_data:
        out.append("  no data  : " + ", ".join(r["label"] for r in no_data))
    out.append("")
    out.append(f"See {opt.slug}_prompt_{run_id}.txt for an analysis prompt "
               f"containing the same data plus the kernel source.")
    out.append("")
    return "\n".join(out)


def build_prompt(opt: Optimization, run_id: str, rows: list[dict]) -> str:
    """Everything needed to analyze the result: intent, source, and numbers."""
    out = [
        "You are a CUDA performance engineer. Analyze the Nsight Compute "
        "counters below against the kernel source that produced them.",
        "",
        "=" * 78,
        "1. WHAT THE OPTIMIZATION CLAIMS",
        "=" * 78,
        opt.summary,
        "",
        "=" * 78,
        "2. COMPLETE SOURCE UNDER TEST",
        "=" * 78,
        f"File: {os.path.relpath(opt.source_path, optimizations.ROOT)}",
        "",
        "```cuda",
        opt.source.rstrip(),
        "```",
        "",
        "=" * 78,
        "3. MEASUREMENT SETUP",
        "=" * 78,
        f"Run id: {run_id}",
        "Each variant runs in its own process, invoked as "
        f"`{opt.slug} before` and `{opt.slug} after`.",
        "Each process performs one warmup launch followed by a timed loop; ncu "
        f"profiles launch index {LAUNCH_SKIP} (the first timed launch) with "
        "--launch-count 1.",
        f"Profiled kernels: {opt.before_kernel} (before), "
        f"{opt.after_kernel} (after).",
        "Both variants are verified to produce equivalent output before timing.",
        "",
        "=" * 78,
        "4. MEASURED COUNTERS",
        "=" * 78,
    ]
    out += metric_table(rows, 78)
    out += ["", "Per-counter detail:", ""]
    for r in rows:
        out.append(f"- {r['label']} ({r['counter']}, unit: {r['unit'] or 'n/a'})")
        out.append(f"    before = {r['before']}, after = {r['after']}, "
                   f"change = {r['change']}")
        out.append(f"    predicted direction = {r['expected']}, "
                   f"observed = {r['verdict']}")
        out.append(f"    why this counter matters: {r['why']}")

    out += [
        "",
        "=" * 78,
        "5. YOUR TASK",
        "=" * 78,
        "1. State whether kernel time improved and by how much, using the "
        "actual numbers.",
        "2. For each counter, explain the hardware mechanism that produced the "
        "observed change, referring to specific lines of the source above.",
        "3. For any counter marked NOT MET, determine the most likely cause. "
        "Consider: the compiler already performed the transformation; a "
        "different bottleneck dominates at this problem size; the working set "
        "fits in cache so the effect is hidden; the launch configuration "
        "changed something else at the same time; or the counter does not "
        "measure what the prediction assumed.",
        "4. State whether the counters, taken together, corroborate the claimed "
        "benefit. Say so plainly if they do not.",
        "5. Propose one concrete next experiment: a specific change to the "
        "source, problem size, or launch configuration, and what counter you "
        "would expect it to move.",
        "",
        "Reference concrete numbers throughout. Do not restate the counter "
        "descriptions back to me. Keep the answer under 400 words.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile before/after CUDA kernel pairs with Nsight Compute.",
        epilog="examples:\n"
               "  %(prog)s\n"
               "  %(prog)s --only 4,8\n"
               "  %(prog)s --only \"memory coalescing,bank\"",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", metavar="LIST",
                        help="comma-separated optimization ids or name "
                             "substrings (default: all twelve)")
    args = parser.parse_args()

    selected = optimizations.select(args.only)

    if not shutil.which("ncu"):
        print("ERROR: 'ncu' not found on PATH.\n"
              "  Try: export PATH=/usr/local/cuda/bin:$PATH", file=sys.stderr)
        sys.exit(1)

    missing = [o for o in selected if not os.path.isfile(o.binary)]
    if missing:
        print("ERROR: binaries not built: "
              + ", ".join(o.slug for o in missing)
              + "\n  Run: make build", file=sys.stderr)
        sys.exit(1)

    # ncu drops unknown counters without complaining, which is indistinguishable
    # from a flat result, so resolve them before spending minutes profiling.
    # Required counters that are missing are an error; optional ones ("?counter"
    # in the .cu header) are architecture-specific and simply skipped.
    available = supported_metrics()
    skipped: set = set()
    if available:
        def missing(m) -> bool:
            return base_counter(m.name) not in available

        unknown = sorted({
            m.name for o in selected for m in o.metrics
            if missing(m) and not m.optional
        })
        if unknown:
            print("ERROR: counters not supported by this GPU:\n  "
                  + "\n  ".join(unknown)
                  + "\n\nFix the @metric lines in src/*.cu. "
                    "List valid names with: ncu --query-metrics",
                  file=sys.stderr)
            sys.exit(1)

        skipped = {m.name for o in selected for m in o.metrics
                   if missing(m) and m.optional}
        if skipped:
            print("Note: optional counters unavailable on this GPU, skipping:\n  "
                  + "\n  ".join(sorted(skipped)) + "\n")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Run id : {run_id}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Running: {len(selected)}/{len(OPTIMIZATIONS)} optimizations\n")

    for opt in selected:
        counters = [m.name for m in opt.metrics if m.name not in skipped]
        reports = {
            v: os.path.join(RESULTS_DIR, f"{opt.slug}_{v}_{run_id}.ncu-rep")
            for v in ("before", "after")
        }

        print(f"  {opt.display} ...", end=" ", flush=True)
        mb = run_ncu(opt.binary, "before", opt.before_kernel, counters,
                     reports["before"])
        ma = run_ncu(opt.binary, "after", opt.after_kernel, counters,
                     reports["after"])
        rows = collect_rows(opt, mb, ma)

        judged = [r for r in rows if r["expected"] != "any" and r["pct"] is not None]
        met = [r for r in judged if r["verdict"] == "MET"]
        print(f"done  ({len(met)}/{len(judged)} expectations met)")

        check_path  = os.path.join(RESULTS_DIR, f"{opt.slug}_check_{run_id}.txt")
        prompt_path = os.path.join(RESULTS_DIR, f"{opt.slug}_prompt_{run_id}.txt")
        with open(check_path, "w", encoding="utf-8") as f:
            f.write(build_check(opt, run_id, rows, reports))
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(build_prompt(opt, run_id, rows))

    print(f"\nArtifacts for run {run_id} are in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
