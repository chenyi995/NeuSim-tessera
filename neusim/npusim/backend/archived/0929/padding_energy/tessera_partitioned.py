"""Partition-aware array, shared-SRAM and reduction counts for native NeuSim.

Timing: gemmini/partition/TIMING_AUDIT.md (array boundary, ingress and seams).
Reuse: Tessera-HPCA2027-revision/sec/04_design.tex, Strip memory and Mapping.
Memory tile search, bandwidths, VU issue cost and power remain native NeuSim.
All arithmetic counts are useful, unpadded operations. See docs/tessera_partitioned.md.
"""
from __future__ import annotations
from contextlib import redirect_stdout
from functools import lru_cache
import json
from math import ceil

from neusim.npusim.backend.tessera import bank_cycles, ceildiv

BACKEND = "tessera_partitioned"
VARIANTS = ("full", "square", "independent", "independent_noskew", "skew", "ws")


class Sink:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def parts(size, tile):
    count, remainder = divmod(size, tile)
    return ([(tile, count)] if count else []) + ([(remainder, 1)] if remainder else [])


def powers(start, end):
    while start <= end:
        yield start
        start *= 2


def parameters(config):
    p = config.tessera_parameters
    grain = int(p.get("grain", 32))  # revision final-three-ablations: grain=32
    side = config.sa_dim
    if config.num_sa != 1 or side < grain or side % grain or side & (side - 1) or grain & (grain - 1):
        raise ValueError("partitioned backend requires one dyadic side x side fabric and dyadic grain")
    if config.tessera_variant not in VARIANTS:
        raise ValueError("unknown partitioned variant")
    return grain, side


def geometries(m, n, k, config):
    grain, side = parameters(config)
    variant = config.tessera_variant
    if variant in ("independent", "independent_noskew"):
        choices = [(grain, grain, True)]
    elif variant == "ws":
        choices = [(side, side, True)]
    else:
        choices = [(kt, nt, square) for kt in powers(grain, side) for nt in powers(grain, side)
                   for square in ((True,) if variant == "square" else ((False, True) if kt != nt else (False,)))]
    seen = set()
    for kt, nt, square in choices:
        kh = max(grain, 1 << (min(k, kt) - 1).bit_length())
        nw = max(grain, 1 << (min(n, nt) - 1).bit_length())
        if variant == "ws":
            kh = nw = side
        elif square:
            kh = nw = max(kh, nw)
        h, w = min(kh, nw), max(kh, nw)
        slots = (side // h) * (side // w)
        for kp in powers(1, min(slots, 1 << (ceildiv(k, kt) - 1).bit_length())):
            value = kt, nt, square, kp, h, w
            if value not in seen:
                seen.add(value)
                yield value


@lru_cache(maxsize=131072)
def memory_tile(m, n, k, a_bytes, b_bytes, accum_bytes, kp, capacity, side, freq, bw):
    """Native factor search with two operand buffers and explicit partial workspace.

    One running FP32 C plane and kp producer planes are live. Streaming reduction
    keeps them on chip; reducing M/N tile sizes reloads A/B from HBM as necessary.
    """
    from neusim.npusim.backend.npusim_lib import find_best_tile_shape_for_matmul
    output_workspace_bytes = (kp + 1) * accum_bytes
    demand = 2 * a_bytes * m * k + 2 * b_bytes * k * n + output_workspace_bytes * m * n
    if demand <= capacity:
        return m, n, k, demand
    result = find_best_tile_shape_for_matmul(
        2 * a_bytes, 2 * b_bytes, output_workspace_bytes, side,
        m, n, k, 1, capacity, 0, freq, bw)
    mt, nt, kt = map(int, result[-3:])
    demand = 2 * a_bytes * mt * kt + 2 * b_bytes * kt * nt + output_workspace_bytes * mt * nt
    assert demand <= capacity
    return mt, nt, kt, demand


def checked_tile(m, n, k, a_bytes, b_bytes, accum, planes, capacity, tile):
    """Capacity for ping-pong A/B, producer planes and resident running C."""
    mt, nt, kt = tile
    if not (0 < mt <= m and 0 < nt <= n and 0 < kt <= k):
        raise ValueError("invalid forced SRAM tile")
    peak = 2*a_bytes*mt*kt + 2*b_bytes*kt*nt + (planes+1)*accum*mt*nt
    if peak > capacity:
        raise ValueError("forced SRAM tile exceeds capacity")
    return peak


@lru_cache(maxsize=262144)
def counts(b, m, n, k, a_bytes, b_bytes, out_bytes, geometry, variant,
           grain, side, capacity, freq, bw, resident=False, forced_tile=None):
    kt, nt, square, kp, h, w = geometry
    # Codex: decision start — FP32 partial sums and ping-pong buffers follow
    # paper Strip memory; HBM sees actual output dtype, never an imported C64 template.
    accum = 4  # paper sec/04_design.tex: 32-bit partial sums, 16-bit operands
    # chenyi9: decision start — jointly selected SRAM tiles use the same counters.
    if forced_tile is None:
        mt, nm, km, peak = memory_tile(m, n, k, a_bytes, b_bytes, accum, kp, capacity, side, freq, bw)
    else:
        mt, nm, km = forced_tile
        peak = checked_tile(m, n, k, a_bytes, b_bytes, accum, kp, capacity, forced_tile)
    # chenyi9: decision end
    slots = (side // h) * (side // w)
    np = slots // kp
    if min(kt, nt, kp, np) <= 0:
        raise ValueError("invalid partition geometry")
    cycles = reads_a = reads_b = partials = scalar_adds = rounds = tail = 0
    for mm, mc in parts(m, mt):
        for nn, nc in parts(n, nm):
            total_k_groups = 0
            for kk, kc in parts(k, km):
                multiplier = b * mc * nc * kc
                nk, nn_regions = ceildiv(kk, kt), ceildiv(nn, nt)
                r = ceildiv(nk, kp) * ceildiv(nn_regions, np)
                # RTL has one registered stage per crossed physical strip and
                # one initial H-cycle transpose fill; both affect the final tail.
                seams = w // grain - 1 if variant not in ("independent", "independent_noskew", "ws") else 0
                traditional = variant in ("independent", "ws")
                time = bank_cycles(mm, h, r, traditional) + w - h + h + seams
                delta = h - 1 if variant == "skew" else 0
                cycles += multiplier * (time + delta)
                tail += multiplier * delta
                rounds += multiplier * r
                reads_a += multiplier * mm * kk * nn_regions
                reads_b += multiplier * kk * nn
                partials += multiplier * mm * nn * nk
                total_k_groups += kc * nk
            scalar_adds += b * mc * nc * mm * nn * (total_k_groups - 1)
    # Output-stationary native tile policy: A reloaded per N tile and B per M
    # tile; partial sums are explicitly resident rather than silently spilled.
    hbm_a = b * m * k * ceildiv(n, nm) * a_bytes
    hbm_b = b * k * n * ceildiv(m, mt) * b_bytes
    hbm_c = b * m * n * out_bytes
    if resident:
        hbm_a = hbm_b = hbm_c = 0
    array_reads = reads_a * a_bytes + reads_b * b_bytes
    partial_writes = partials * accum
    reduction_reads = scalar_adds * 2 * accum
    reduction_writes = scalar_adds * accum
    # Final accumulators are read for output conversion/drain. In resident
    # attention, the next stage consumes this output in shared SRAM.
    final_reads = b * m * n * accum
    sram = array_reads + partial_writes + reduction_reads + reduction_writes + final_reads + hbm_a + hbm_b
    macs = b * m * n * k
    assert cycles * side * side >= macs
    return dict(model=BACKEND, engine="SA", geometry=list(geometry), variant=variant,
                B=b, M=m, N=n, K=k, useful_macs=macs, sa_cycles=cycles,
                sa_arithmetic_ops=2*macs,
                reduction_ops=scalar_adds, vu_arithmetic_ops=scalar_adds,
                array_a_read_bytes=reads_a*a_bytes, array_b_read_bytes=reads_b*b_bytes,
                partial_write_bytes=partial_writes, reduction_read_bytes=reduction_reads,
                reduction_write_bytes=reduction_writes, final_read_bytes=final_reads,
                sram_bytes=sram, hbm_a_read_bytes=hbm_a, hbm_b_read_bytes=hbm_b,
                hbm_output_write_bytes=hbm_c, hbm_partial_read_bytes=0, hbm_partial_write_bytes=0,
                hbm_bytes=hbm_a+hbm_b+hbm_c, peak_live_bytes=peak,
                memory_tile=[mt, nm, km], rounds=rounds, skew_tail_cycles=tail)
    # Codex: decision end


def issue_ns(scalar_ops, config):
    from neusim.npusim.backend.npusim_lib import compute_node_cost_vpu_time_from_num_ops
    # Native VU has 8 sublanes x 128 lanes; reduction is one scalar add per lane.
    return compute_node_cost_vpu_time_from_num_ops(ceildiv(scalar_ops, 8 * 128), config) if scalar_ops else 0


def evaluate_candidate(b, m, n, k, dtype, geometry, config, resident=False, forced_tile=None):
    from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
    from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
    op = create_einsum_op([b, m, k], [b, k, n], "BMK;BKN->BMN", dtype=dtype)
    op.tessera_spec = dict(kind="native_partition", geometry=list(geometry), resident=resident)
    if forced_tile is not None:
        op.tessera_spec["memory_tile"] = list(forced_tile)
    with redirect_stdout(Sink()):
        return fill_operators_execution_info([op], config)[0]


@lru_cache(maxsize=65536)
def select_geometry(b, m, n, k, dtype, config_json, resident=False):
    from neusim.configs.chips.ChipConfig import ChipConfig
    config = ChipConfig.model_validate_json(config_json)
    # Skew changes only final drain, fixing both array and native memory mapping.
    if config.tessera_variant == "skew":
        other = config.model_copy(update={"tessera_variant": "full"})
        return select_geometry(b, m, n, k, dtype, other.model_dump_json(), resident)
    best = None
    # chenyi9: decision start — evaluate paper baselines through native NeuSim.
    if config.tessera_parameters.get("baseline_architecture"):
        from neusim.npusim.backend.tessera_baselines import geometries as baseline_geometries
        candidates = baseline_geometries(config)
    else:
        candidates = geometries(m, n, k, config)
    # chenyi9: decision end
    for geometry in candidates:
        op = evaluate_candidate(b, m, n, k, dtype, geometry, config, resident)
        s = op.stats
        key = (s.total_energy_J * s.execution_time_ns, s.execution_time_ns, geometry)
        if best is None or key < best[0]:
            best = key, geometry
    if best is None:
        raise ValueError("no feasible partitioned candidate")
    return best[1]


def compute_matmul(I, op, config, b, m, n, k):
    from neusim.npusim.backend import util
    from neusim.npusim.backend.npusim_lib import compute_node_cost_vpu_time_from_num_ops
    grain, side = parameters(config)
    spec = op.tessera_spec or {}
    geometry = spec.get("geometry")
    forced_tile = tuple(spec["memory_tile"]) if "memory_tile" in spec else None
    dtype = op.input_tensors[0].dtype
    # Tensor dtype is a native string; use the HLO scalar spelling if needed.
    if geometry is None:
        # chenyi9: decision start — all architectures share joint EDP selection.
        if config.tessera_parameters.get("mapping_policy") == "joint_edp":
            from neusim.npusim.backend.tessera_joint import select_mapping
            geometry, forced_tile = select_mapping(b, m, n, k, dtype, config.model_dump_json(), spec.get("resident", False))
        else:
            geometry = select_geometry(b, m, n, k, dtype, config.model_dump_json(), spec.get("resident", False))
        # chenyi9: decision end
    geometry = tuple(geometry)
    baseline = config.tessera_parameters.get("baseline_architecture")
    if baseline:
        from neusim.npusim.backend.tessera_baselines import geometries as baseline_geometries
        legal = baseline_geometries(config)
    else:
        legal = geometries(m, n, k, config)
    if geometry not in set(legal):
        raise ValueError("forced partition geometry is not legal for this shape/variant")
    ab = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    bb = util.get_size_bytes_from_dtype(I.input_axes[1][0].data_type)
    cb = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)
    if baseline:
        from neusim.npusim.backend.tessera_baselines import counts as baseline_counts
        detail = dict(baseline_counts(b, m, n, k, ab, bb, cb, geometry, baseline,
            config.vmem_size_MB*1024**2, config.freq_GHz, config.hbm_bw_GBps, spec.get("resident", False), forced_tile))
    else:
        detail = dict(counts(b, m, n, k, ab, bb, cb, geometry, config.tessera_variant,
                            grain, side, config.vmem_size_MB*1024**2, config.freq_GHz,
                            config.hbm_bw_GBps, spec.get("resident", False), forced_tile))
    sa, vu = ceil(detail["sa_cycles"] / config.freq_GHz), issue_ns(detail["reduction_ops"], config)
    # Preserve native VU matmul issue formula and its four-times selection policy.
    native_vu_ops = b * min(ceildiv(m, 8)*ceildiv(n, 128)*ceildiv(k, 8),
                            ceildiv(m, 128)*ceildiv(n, 8)*ceildiv(k, 8),
                            ceildiv(m, 8)*ceildiv(n, 8)*ceildiv(k, 128)) * 2
    alternate = compute_node_cost_vpu_time_from_num_ops(native_vu_ops, config)
    if config.use_vu_for_small_matmul and max(sa, vu) > 4 * alternate:
        detail.update(engine="VU", sa_arithmetic_ops=0, vu_arithmetic_ops=2*b*m*n*k)
        # Codex: retain the rejected mapping for diagnosis, but no executed
        # array traffic or regional reductions may be attributed to VU work.
        detail["rejected_sa_cycles"] = detail["sa_cycles"]
        for key in ("sa_cycles", "rounds", "reduction_ops", "array_a_read_bytes", "array_b_read_bytes",
                    "partial_write_bytes", "reduction_read_bytes", "reduction_write_bytes",
                    "final_read_bytes", "skew_tail_cycles"):
            detail[key] = 0
        sa, vu = 0, alternate
    op.stats.tessera_details = detail
    op.stats.num_sa_ops = detail["rounds"] if sa else 0
    op.stats.einsum_B_size, op.stats.einsum_M_size = b, m
    op.stats.einsum_N_size, op.stats.einsum_K_size = n, k
    return sa, vu


@lru_cache(maxsize=65536)
def attention_counts(batch, q, kv, hq, hkv, d, dtype, config_json):
    from neusim.configs.chips.ChipConfig import ChipConfig
    from neusim.npusim.frontend.llm_ops_lib import create_multi_head_flash_attention_op
    from neusim.npusim.backend import npusim_lib, util
    config = ChipConfig.model_validate_json(config_json)
    grain, side = parameters(config)
    if hq % hkv:
        raise ValueError("GQA requires an integral query-head group")
    group = hq // hkv
    op = create_multi_head_flash_attention_op([batch, q, hq, d], [batch, kv, hkv, d], [batch, kv, hkv, d], dtype=dtype)
    module = util.construct_hlo_module_from_node_costs([op])
    I, op = npusim_lib.parse_tensor_shapes_for_node_cost(op, module)
    bc, br = npusim_lib.get_best_tile_config_for_flash_attention_from_vmem_size(I, op, config)
    br, bc = min(br, q), min(bc, kv)
    word = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    capacity = config.vmem_size_MB * 1024**2

    def working_set(br, bc):
        # Codex: decision start — retain native FlashAttention loop order and
        # reserve Q/K/V ping-pong, score buffers, FP32 output, and one wave of
        # regional partials. At most side^2/grain output lanes can drain at once.
        m = br * group
        return (2 * word * (m*d + 2*bc*d) + 2*4*m*bc
                + 4*m*d + 4*m*(side*side//grain))
        # Codex: decision end

    while working_set(br, bc) > capacity:
        if bc >= br and bc > 1:
            bc = ceildiv(bc, 2)
        elif br > 1:
            br = ceildiv(br, 2)
        else:
            raise ValueError("attention working set cannot fit SRAM")
    summed = dict.fromkeys(("sa_cycles", "reduction_ops", "vu_arithmetic_ops", "useful_macs",
                           "array_a_read_bytes", "array_b_read_bytes", "partial_write_bytes",
                           "reduction_read_bytes", "reduction_write_bytes", "final_read_bytes",
                           "sram_bytes", "rounds", "skew_tail_cycles"), 0)
    phase_plans = []
    executed_sa_ops = 0
    # Materialize only run-length groups of edge tiles; independent KV groups
    # execute serially and share their K/V across the grouped query heads.
    for qr, qr_count in parts(q, br):
        for kc, kc_count in parts(kv, bc):
            copies = batch * hkv * qr_count * kc_count
            m = qr * group
            for label, nn, kk in (("QK", kc, d), ("PV", d, kc)):
                # chenyi9: decision start — resident attention phases share the mapper.
                tile = None
                if config.tessera_parameters.get("mapping_policy") == "joint_edp":
                    from neusim.npusim.backend.tessera_joint import select_mapping
                    geometry, tile = select_mapping(1, m, nn, kk, dtype, config_json, True)
                else:
                    geometry = select_geometry(1, m, nn, kk, dtype, config_json, True)
                phase = evaluate_candidate(1, m, nn, kk, dtype, geometry, config, resident=True, forced_tile=tile)
                # chenyi9: decision end
                info = phase.stats.tessera_details
                executed_sa_ops += copies * info["sa_arithmetic_ops"]
                # Fused QK and PV must share the selected engine policy. Store
                # native selected active times separately when fallback is enabled.
                for key in summed:
                    summed[key] += copies * info[key]
                phase_plans.append(dict(phase=label, M=m, N=nn, K=kk, copies=copies,
                                        geometry=list(geometry), engine=info["engine"],
                                        memory_tile=info["memory_tile"],
                                        sa_ns=phase.stats.sa_time_ns, vu_ns=phase.stats.vu_time_ns))
    softmax_ops = 4 * batch * hq * q * kv  # native Softmax scalar-operation count
    combine_ops = batch * hq * q * d * (ceildiv(kv, bc) - 1)
    summed["reduction_ops"] += combine_ops
    summed["vu_arithmetic_ops"] += softmax_ops + combine_ops
    summed["reduction_read_bytes"] += combine_ops * 8
    summed["reduction_write_bytes"] += combine_ops * 4
    # Native softmax reads/writes score tiles; intermediate QK/PV tensors stay
    # in SRAM. HBM traffic comes from the native FlashAttention helper below.
    softmax_sram = batch * hq * q * kv * word * 2
    hbm = npusim_lib.compute_bytes_accessed_from_tile_config_flash_attention(I, op, bc, br)
    summed["sram_bytes"] += combine_ops * 12 + softmax_sram + hbm
    assert summed["useful_macs"] == 2 * batch * hq * q * kv * d
    summed.update(model=BACKEND, engine="SA", variant=config.tessera_variant,
                  sa_arithmetic_ops=executed_sa_ops, hbm_bytes=hbm,
                  hbm_partial_read_bytes=0, hbm_partial_write_bytes=0,
                  peak_live_bytes=working_set(br, bc), memory_tile=[br, bc],
                  phase_plans=phase_plans, softmax_ops=softmax_ops,
                  combine_ops=combine_ops, Q=q, KV=kv, Hq=hq, Hkv=hkv, D=d, B=batch)
    # All-SA primary uses cycle counts before native energy analysis. The
    # optional fallback case includes the per-phase native engine decision.
    if not config.use_vu_for_small_matmul:
        sa_ns = ceil(summed["sa_cycles"] / config.freq_GHz)
        vu_ns = issue_ns(summed["vu_arithmetic_ops"], config)
    else:
        sa_ns = sum(x["copies"] * x["sa_ns"] for x in phase_plans)
        vu_ns = sum(x["copies"] * x["vu_ns"] for x in phase_plans) + issue_ns(softmax_ops + combine_ops, config)
    return summed, sa_ns, vu_ns


def compute_attention(I, op, config):
    from neusim.npusim.backend.npusim_lib import get_axes_size_for_flash_attention
    b, q, kv, hq, hkv, d = get_axes_size_for_flash_attention(I, op)
    detail, sa, vu = attention_counts(b, q, kv, hq, hkv, d, op.input_tensors[0].dtype, config.model_dump_json())
    op.stats.tessera_details = dict(detail)
    op.stats.num_sa_ops = detail["rounds"]
    op.stats.vu_softmax_time_ns = issue_ns(detail["softmax_ops"], config)
    op.stats.einsum_B_size = b * hkv
    op.stats.einsum_M_size, op.stats.einsum_N_size, op.stats.einsum_K_size = q*(hq//hkv), kv, d
    return sa, vu
