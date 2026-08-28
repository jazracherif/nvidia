"""
Optimization catalog, parsed from the benchmark sources.

Each src/NN_name.cu carries a structured header comment that is the single
source of truth for the optimization: its description and the ncu counters that
judge it.  Nothing here is hardcoded, so the .cu file and the profiling report
cannot drift apart.

Header format (tags may wrap onto continuation lines):

    /**
     * @id        4
     * @name      Memory Coalescing
     * @benefit   ...
     * @strategy  ...
     * @algorithm ...
     * @before    ...
     * @after     ...
     * @kernel_before memory_coalescing_before_kernel
     * @kernel_after  memory_coalescing_after_kernel
     *
     * @metric <ncu counter> | <column label> | <up|down|any> | <why it matters>
     */

A counter prefixed with '?' is optional: collected where the device supports it,
skipped elsewhere.  Use it for counters that exist only on some architectures,
such as dram__* on discrete GPUs.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

ROOT      = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(ROOT, "src")
BUILD_DIR = os.path.join(ROOT, "build")

EXPECT_VALUES = ("up", "down", "any")

_TAG_RE    = re.compile(r"^@(\w+)\s*(.*)$")
_HEADER_RE = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)


class CatalogError(Exception):
    """A .cu header is missing or malformed."""


@dataclass(frozen=True)
class Metric:
    """One ncu counter as read in the context of one optimization."""

    name: str      # ncu perfworks counter, e.g. "dram__bytes_read.sum"
    label: str     # short column header, e.g. "DRAMRead(B)"
    expect: str    # "up", "down", or "any" for informational only
    why: str       # what it measures and why it reveals this optimization
    # Counters that only exist on some architectures are written "?counter" and
    # are dropped on devices that lack them instead of failing validation.
    optional: bool = False


@dataclass(frozen=True)
class Optimization:
    """A before/after kernel pair, its rationale, and the counters that judge it."""

    id: int
    name: str
    slug: str                 # source stem, e.g. "04_memory_coalescing"
    source_path: str
    source: str               # full .cu text, embedded verbatim in prompts
    benefit: str
    strategy: str
    algorithm: str
    before_desc: str
    after_desc: str
    before_kernel: str
    after_kernel: str
    metrics: tuple[Metric, ...]

    @property
    def display(self) -> str:
        """'4. Memory Coalescing'"""
        return f"{self.id}. {self.name}"

    @property
    def binary(self) -> str:
        return os.path.join(BUILD_DIR, self.slug)

    @property
    def summary(self) -> str:
        """Prose description, without the source listing."""
        return "\n".join([
            f"Optimization: {self.display}",
            f"Claimed benefit: {self.benefit}",
            f"Strategy: {self.strategy}",
            f"Algorithm under test: {self.algorithm}",
            "",
            f"Before ({self.before_kernel}): {self.before_desc}",
            f"After  ({self.after_kernel}): {self.after_desc}",
        ])


def _split_tags(block: str, path: str) -> tuple[dict, list[str]]:
    """Split a header comment into {tag: value} plus the ordered @metric lines."""
    tags: dict[str, str] = {}
    metrics: list[str] = []
    current: Optional[str] = None  # tag name, or "metric" for the last metric

    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("*"):
            line = line[1:].strip()

        match = _TAG_RE.match(line)
        if match:
            tag, value = match.group(1), match.group(2).strip()
            if tag == "metric":
                metrics.append(value)
                current = "metric"
            else:
                if tag in tags:
                    raise CatalogError(f"{path}: duplicate @{tag}")
                tags[tag] = value
                current = tag
        elif line and current:
            # Continuation of the previous tag.
            if current == "metric":
                metrics[-1] += " " + line
            else:
                tags[current] += " " + line
        elif not line:
            current = None  # blank line ends a tag

    return tags, metrics


def _parse_metric(spec: str, path: str) -> Metric:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 4:
        raise CatalogError(
            f"{path}: @metric needs 4 pipe-separated fields "
            f"(counter | label | expect | why), got {len(parts)}: {spec!r}")
    name, label, expect, why = parts
    optional = name.startswith("?")
    name = name.lstrip("?").strip()
    if expect not in EXPECT_VALUES:
        raise CatalogError(
            f"{path}: @metric '{label}' has expect={expect!r}, "
            f"must be one of {EXPECT_VALUES}")
    return Metric(name=name, label=label, expect=expect, why=why,
                  optional=optional)


def parse_source(path: str) -> Optimization:
    """Build an Optimization from one .cu file's header comment."""
    with open(path, encoding="utf-8") as f:
        source = f.read()

    header = _HEADER_RE.search(source)
    if not header:
        raise CatalogError(f"{path}: no /** ... */ header comment found")

    tags, metric_specs = _split_tags(header.group(1), path)

    required = ("id", "name", "kernel_before", "kernel_after")
    missing = [t for t in required if t not in tags]
    if missing:
        raise CatalogError(f"{path}: missing required tag(s): "
                           + ", ".join("@" + t for t in missing))
    if not metric_specs:
        raise CatalogError(f"{path}: no @metric lines")

    if not tags["id"].isdigit():
        raise CatalogError(f"{path}: @id must be an integer, got {tags['id']!r}")

    metrics = tuple(_parse_metric(spec, path) for spec in metric_specs)
    labels = [m.label for m in metrics]
    if len(set(labels)) != len(labels):
        raise CatalogError(f"{path}: duplicate @metric labels: {labels}")

    for role, symbol in (("kernel_before", tags["kernel_before"]),
                         ("kernel_after", tags["kernel_after"])):
        if not re.search(rf"__global__[^;{{]*\b{re.escape(symbol)}\s*\(", source):
            raise CatalogError(
                f"{path}: @{role} '{symbol}' is not defined as a __global__ "
                f"kernel in this file")

    return Optimization(
        id=int(tags["id"]),
        name=tags["name"],
        slug=os.path.splitext(os.path.basename(path))[0],
        source_path=path,
        source=source,
        benefit=tags.get("benefit", ""),
        strategy=tags.get("strategy", ""),
        algorithm=tags.get("algorithm", ""),
        before_desc=tags.get("before", ""),
        after_desc=tags.get("after", ""),
        before_kernel=tags["kernel_before"],
        after_kernel=tags["kernel_after"],
        metrics=metrics,
    )


def load() -> list[Optimization]:
    """Parse every src/*.cu, ordered by id."""
    paths = sorted(
        os.path.join(SRC_DIR, f)
        for f in os.listdir(SRC_DIR) if f.endswith(".cu")
    ) if os.path.isdir(SRC_DIR) else []

    if not paths:
        raise CatalogError(f"no .cu sources found in {SRC_DIR}")

    opts = [parse_source(p) for p in paths]

    by_id: dict[int, Optimization] = {}
    for opt in opts:
        if opt.id in by_id:
            raise CatalogError(
                f"duplicate @id {opt.id} in {by_id[opt.id].slug} and {opt.slug}")
        by_id[opt.id] = opt

    return sorted(opts, key=lambda o: o.id)


OPTIMIZATIONS: list[Optimization] = load()


def select(spec: Optional[str]) -> list[Optimization]:
    """Filter OPTIMIZATIONS by comma-separated ids or name substrings."""
    if not spec:
        return OPTIMIZATIONS

    selected: list[Optimization] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        matches = [
            opt for opt in OPTIMIZATIONS
            if (token.isdigit() and opt.id == int(token))
            or (not token.isdigit() and token in opt.name.lower())
        ]
        if not matches:
            raise SystemExit(f"ERROR: no optimization matches '{raw.strip()}'.\n"
                             "  Available: "
                             + ", ".join(o.display for o in OPTIMIZATIONS))
        for m in matches:
            if m not in selected:
                selected.append(m)
    return selected


if __name__ == "__main__":
    for o in OPTIMIZATIONS:
        print(f"{o.slug:<28} {o.display:<26} {len(o.metrics)} metrics")
        for m in o.metrics:
            print(f"    {m.label:<18} {m.expect:<5} {m.name}")
