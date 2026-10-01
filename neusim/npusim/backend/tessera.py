"""Analytical equal-PE Tessera ablations; all dimensional costs are explicit.

Timing reference: gemmini/partition/TIMING_AUDIT.md, array boundary; banking:
FissionSA/fissionsa/revision_arrays.py::_bank_cycles. These are references,
not runtime imports. See docs/tessera.md for the accounting boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from math import ceil


VARIANTS = ("tessera", "square", "independent", "skew", "ws")


@dataclass(frozen=True)
class TesseraConfig:
    side: int
    grain: int
    frequency_hz: float
    sram_bytes: int
    staging_bytes: int
    operand_bytes: int
    accumulator_bytes: int
    hbm_bytes_per_second: float
    hbm_read_pj_per_byte: float
    hbm_write_pj_per_byte: float
    array_pj_per_op: float
    sram_read_pj_per_operand: float
    sram_write_pj_per_accumulator: float
    vector_ops_per_cycle: float
    vector_pj_per_op: float
    background_power_W: float
    seam_cycles: int
    ingress_cycles_per_row: int
    ici_bytes_per_second: float
    ici_latency_ns: float
    ici_pj_per_byte: float

    def __post_init__(self):
        for name in ("side", "grain", "sram_bytes", "staging_bytes",
                     "operand_bytes", "accumulator_bytes"):
            x = getattr(self, name)
            if not isinstance(x, int) or x <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.side % self.grain or self.side & (self.side - 1) or self.grain & (self.grain - 1):
            raise ValueError("side and grain must be powers of two, grain <= side")
        if self.sram_bytes <= self.staging_bytes:
            raise ValueError("SRAM must exceed transfer staging")
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"negative {name}")
        if min(self.frequency_hz, self.hbm_bytes_per_second, self.vector_ops_per_cycle,
               self.ici_bytes_per_second) <= 0:
            raise ValueError("frequency, throughput and bandwidth must be positive")

    @property
    def pe_count(self):
        return self.side ** 2


def ceildiv(a, b):
    return (a + b - 1) // b


def bank_cycles(m: int, h: int, folds: int, skew: bool) -> int:
    """Exact final-output time for the shared two-weight-bank recurrence.

    Load F[f]=max(F[f-1]+h,I[f-2]+m+release), I[f]=max(F[f]+h,I[f-1]+m).
    Source: revision_arrays.py::_bank_cycles; an independent event reference
    lives in tests/test_tessera.py. A MAC is two arithmetic operations.
    """
    if min(m, h, folds) <= 0:
        raise ValueError("nonpositive array extent")
    release = h - 1 if skew else 0
    drain = 2 * h - 2 if skew else h - 1
    a = max(m, h)
    b = m + release + h
    last = folds - 1
    return h + last * a + (last // 2) * max(0, b - 2 * a) + m + drain


def _physical_extent(actual, grain):
    # Codex: decision start — use dyadic rectangles so every mirror chain is integral.
    return max(grain, 1 << (actual - 1).bit_length())
    # Codex: decision end


@dataclass(frozen=True)
class ArrayResult:
    cycles: int
    kt: int
    nt: int
    physical_h: int
    physical_w: int
    rounds: int
    useful_macs: int
    input_words: int
    partial_output_words: int
    reduction_ops: int
    seam_tail_cycles: int
    skew_extra_cycles: int
    ingress_cycles: int


@lru_cache(maxsize=65536)
def array_cost(m: int, n: int, k: int, batch: int, variant: str,
               kt: int, nt: int, side: int, grain: int,
               seam_cycles: int = 1, ingress_cycles_per_row: int = 1) -> ArrayResult:
    """Uniform slots including tails; never issue padded operands or MACs.

    All rounds retain fixed physical slots. A logical remainder occupies the
    same slot as the full tile; no unsupported heterogeneous bank reuse is
    assumed. This is deliberately a restricted, reproducible mapper.
    """
    if variant not in VARIANTS or min(m, n, k, batch, kt, nt, side, grain) <= 0:
        raise ValueError("invalid array query")
    if max(kt, nt) > side or side % kt or side % nt or min(kt, nt) < grain:
        raise ValueError("illegal tile")
    if variant == "independent" and (kt, nt) != (grain, grain):
        raise ValueError("independent arrays cannot merge")
    if variant == "ws" and (kt, nt) != (side, side):
        raise ValueError("monolithic WS has one physical array")
    kh = _physical_extent(min(k, kt), grain)
    nw = _physical_extent(min(n, nt), grain)
    if variant == "ws":
        kh = nw = side
    elif variant == "square":
        kh = nw = max(kh, nw)
    h, w = min(kh, nw), max(kh, nw)
    capacity = (side // h) * (side // w)
    tiles = batch * ceildiv(k, kt) * ceildiv(n, nt)
    rounds = ceildiv(tiles, capacity)
    is_skew = variant in ("skew", "ws")
    base = bank_cycles(m, h, rounds, False) + w - h
    cycles = bank_cycles(m, h, rounds, is_skew) + w - h
    # chenyi9: decision start — calibrate registered strip seams and transpose fill to Gemmini.
    seams = (w // grain - 1) * seam_cycles if variant in ("tessera", "square", "skew") else 0
    ingress = h * ingress_cycles_per_row
    # chenyi9: decision end
    inwords = batch * (m * k * ceildiv(n, nt) + k * n)
    partials = batch * m * n * ceildiv(k, kt)
    reductions = batch * m * n * (ceildiv(k, kt) - 1)
    return ArrayResult(cycles + seams + ingress, kt, nt, h, w, rounds,
                       batch * m * n * k, inwords, partials, reductions,
                       seams, cycles - base, ingress)


def candidates(c: TesseraConfig, variant: str):
    if variant == "independent":
        return [(c.grain, c.grain)]
    if variant == "ws":
        return [(c.side, c.side)]
    sizes = []
    d = c.grain
    while d <= c.side:
        sizes.append(d)
        d *= 2
    return [(k, n) for k in sizes for n in sizes]


@dataclass(frozen=True)
class MemoryPlan:
    m_chunk: int
    n_chunk: int
    chunks: tuple[tuple[int, int, int, int], ...]
    b_loads: int
    peak_live_bytes: int
    strategy: str


@lru_cache(maxsize=8192)
def plan_memory(m: int, n: int, k: int, c: TesseraConfig, batch: int = 1) -> MemoryPlan:
    """Retain A rows and FP32 C through all K tiles; streamed or retained B.

    Chunks are (rows, columns, multiplicity, B read bytes per occurrence).
    SRAM reads by the array and HBM transfers are distinct counters.
    Reference: revision_memory.py::plan_memory; this implementation does not
    import it and never assumes a fitting tensor was already loaded.
    """
    budget = (c.sram_bytes - c.staging_bytes) // batch
    widths = {n, min(n, c.side)}
    x = c.side
    while x < n:
        widths.add(x)
        x *= 2
    choices = []
    for retained in (False, True):
        reserve_b = k * n * c.operand_bytes if retained else 0
        for width in sorted(widths):
            mr = min(m, (budget - reserve_b) // (k * c.operand_bytes + width * c.accumulator_bytes))
            if mr < 1:
                continue
            nm, nn = ceildiv(m, mr), ceildiv(n, width)
            bloads = 1 if retained else nm
            # Byte-optimal shared memory policy, as in the revision reference.
            choices.append(((bloads * k * n, nm * nn, -mr, -width), mr, width, retained))
    if not choices:
        raise ValueError("one A row and one output tile do not fit the SRAM budget")
    _, mr, width, retained = min(choices)
    chunks = []
    row_groups = [(mr, m // mr)] + ([(m % mr, 1)] if m % mr else [])
    col_groups = [(width, n // width)] + ([(n % width, 1)] if n % width else [])
    for rows, nr in row_groups:
        for cols, nc in col_groups:
            if nr and nc:
                chunks.append((rows, cols, nr * nc, 0 if retained else k * cols * c.operand_bytes))
    peak = batch * (mr * k * c.operand_bytes + mr * width * c.accumulator_bytes
                    + (k * n * c.operand_bytes if retained else 0)) + c.staging_bytes
    assert peak <= c.sram_bytes
    return MemoryPlan(mr, width, tuple(chunks), 1 if retained else ceildiv(m, mr),
                      peak, "retain_b" if retained else "stream_b")


@dataclass(frozen=True)
class GemmResult:
    seconds: float
    array_cycles: int
    kt: int
    nt: int
    rounds: int
    useful_macs: int
    reduction_ops: int
    array_energy_J: float
    sram_energy_J: float
    vector_energy_J: float
    background_energy_J: float
    hbm_reload_bytes: int
    hbm_energy_J: float
    sram_read_bytes: int
    sram_write_bytes: int
    peak_live_bytes: int
    skew_extra_cycles: int
    seam_tail_cycles: int

    @property
    def energy_J(self):
        return (self.array_energy_J + self.sram_energy_J + self.vector_energy_J
                + self.background_energy_J + self.hbm_energy_J)


@lru_cache(maxsize=65536)
def gemm_cost(m: int, n: int, k: int, batch: int, variant: str,
              c: TesseraConfig, forced_tile: tuple[int, int] | None = None,
              transfer_bytes: int = 0, write_bytes: int = 0) -> GemmResult:
    """Select EDP with the same transfers used by timing and energy.

    Compulsory graph-level bytes are provided by the caller; extra B reloads
    arise here. HBM and compute overlap within the operator (NeuSim roofline
    approximation), bounded below by the first load and last store elsewhere.
    """
    plan = plan_memory(m, n, k, c, batch)
    reloads = batch * (plan.b_loads - 1) * k * n * c.operand_bytes
    hbme = ((transfer_bytes + reloads) * c.hbm_read_pj_per_byte
            + write_bytes * c.hbm_write_pj_per_byte) * 1e-12
    best = None
    for kt, nt in [forced_tile] if forced_tile else candidates(c, variant):
        cycles = inwords = partials = reductions = rounds = skew = seams = 0
        for rows, cols, repeats, _ in plan.chunks:
            a = array_cost(rows, cols, k, batch, variant, kt, nt, c.side, c.grain,
                           c.seam_cycles, c.ingress_cycles_per_row)
            cycles += repeats * a.cycles
            inwords += repeats * a.input_words
            partials += repeats * a.partial_output_words
            reductions += repeats * a.reduction_ops
            rounds += repeats * a.rounds
            skew += repeats * a.skew_extra_cycles
            seams += repeats * a.seam_tail_cycles
        macs = batch * m * n * k
        # Codex: decision start — count every cross-K FP32 accumulation explicitly.
        red_read_bytes = reductions * 2 * c.accumulator_bytes
        red_write_bytes = reductions * c.accumulator_bytes
        sram_reads = inwords * c.operand_bytes + red_read_bytes
        sram_writes = partials * c.accumulator_bytes + red_write_bytes
        sr_energy = (sram_reads / c.operand_bytes * c.sram_read_pj_per_operand
                     + sram_writes / c.accumulator_bytes * c.sram_write_pj_per_accumulator) * 1e-12
        vu_time = reductions / c.vector_ops_per_cycle / c.frequency_hz
        compute_time = cycles / c.frequency_hz + vu_time
        seconds = overlapped_seconds(compute_time, transfer_bytes + reloads, write_bytes, c)
        # Codex: decision end
        result = GemmResult(seconds, cycles, kt, nt, rounds, macs, reductions,
                            2 * macs * c.array_pj_per_op * 1e-12, sr_energy,
                            reductions * c.vector_pj_per_op * 1e-12,
                            seconds * c.background_power_W, reloads, hbme,
                            sram_reads, sram_writes, plan.peak_live_bytes, skew, seams)
        key = (result.energy_J * seconds, seconds, kt, nt)
        if best is None or key < best[0]:
            best = key, result
    return best[1]


def overlapped_seconds(compute, reads, writes, c):
    # Codex: decision start — share the same first-load/last-store bound with EDP selection.
    startup = min(reads, c.staging_bytes // 2) / c.hbm_bytes_per_second
    finish = min(writes, c.staging_bytes // 2) / c.hbm_bytes_per_second
    transfer = (reads + writes) / c.hbm_bytes_per_second
    return startup + max(compute, max(0, transfer - startup - finish)) + finish
    # Codex: decision end
