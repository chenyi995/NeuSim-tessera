"""NeuSim operator adapter and capacity-aware tensor accounting for Tessera."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict
from math import ceil, prod

from neusim.npusim.backend.tessera import TesseraConfig, gemm_cost, plan_memory, overlapped_seconds
from neusim.npusim.frontend.Operator import OpcodeType, OpType


class TensorStore:
    """Whole-tensor LRU at operator boundaries; dirty live evictions go to HBM.

    Array workspace is reserved before execution. Streaming operands remain
    backed by HBM. Values without tensor identity are never presumed resident.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()

    @property
    def size(self):
        return sum(v[0] for v in self.items.values())

    def reserve(self, bytes_available, keep=()):
        writes = 0
        for key in list(self.items):
            if self.size <= bytes_available:
                break
            if key in keep:
                continue
            size, dirty = self.items.pop(key)
            if dirty:
                writes += size
        return writes

    def has(self, name, size):
        if name in self.items and self.items[name][0] == size:
            self.items.move_to_end(name)
            return True
        return False

    def put(self, name, size, dirty):
        if name in self.items:
            self.items.pop(name)
        writes = self.reserve(max(0, self.capacity - size))
        if size <= self.capacity:
            self.items[name] = (size, dirty)
        elif dirty:
            writes += size
        return writes


def _native_spec(op, chip):
    """Convert existing NeuSim operators; no invented inter-op tensor aliasing."""
    from neusim.npusim.backend import npusim_lib, util
    if op.tessera_spec is not None:
        return op.tessera_spec
    hlo = util.construct_hlo_module_from_node_costs([op])
    instruction, _ = npusim_lib.parse_tensor_shapes_for_node_cost(op, hlo)
    inputs = {t.uuid: prod(t.shape) * util.get_size_bytes_from_dtype(t.dtype) for t in op.input_tensors}
    outputs = {t.uuid: prod(t.shape) * util.get_size_bytes_from_dtype(t.dtype) for t in op.output_tensors}
    spec = dict(inputs=inputs, outputs=outputs, final=True, identity="native-cold")
    if op.opcode_type == OpcodeType.EINSUM:
        axes = npusim_lib.separate_axes_by_type_for_matmul(
            *instruction.input_axes, instruction.output_axes)
        ba, ra, ma, na = axes
        spec.update(kind="gemm", batch=prod(a.size for a in ba),
                    M=prod(a.size for a in ma), N=prod(a.size for a in na),
                    K=prod(a.size for a in ra))
    elif op.opcode_type == OpcodeType.FLASH_ATTENTION:
        b, q, kv, hq, hkv, d = npusim_lib.get_axes_size_for_flash_attention(instruction, op)
        spec.update(kind="attention", batch=b, q=q, kv=kv, hq=hq, hkv=hkv, d=d)
    elif op.opcode_type == OpcodeType.CONV2D:
        raise NotImplementedError("Use explicit grouped/Conv GEMM lowering for the Tessera backend")
    else:
        spec.update(kind="vector", vector_ops=op.stats.flop_count,
                    ici_bytes=op.stats.ici_traffic_bytes)
    return spec


def attention_cost(spec, c, variant, fixed_variant=None):
    """Fused GQA attention, with explicit QK -> softmax -> PV dependencies.

    Query heads sharing a KV head are folded into M. Requests stay separate.
    KV blocks retain scores in SRAM; no score tensor is written to HBM.
    Online normalization updates cost multiply/add operations on running
    FP32 output and scalar sums. Exp/div use NeuSim's scalar-op approximation.
    """
    q, kv, hq, hkv, d = (spec[k] for k in ("q", "kv", "hq", "hkv", "d"))
    b = spec.get("batch", 1)
    if hq % hkv or min(q, kv, hq, hkv, d, b) <= 0:
        raise ValueError("invalid grouped-head attention")
    groups = hq // hkv
    qb = min(q, c.side)
    # Reserve Q and FP32 output plus transfer staging; score storage is FP32.
    base = b * qb * hq * d * (c.operand_bytes + c.accumulator_bytes)
    per_k = b * (2 * hkv * d * c.operand_bytes + qb * hq * c.accumulator_bytes)
    kb = min(kv, (c.sram_bytes - c.staging_bytes - base) // per_k)
    while kb < 1 and qb > 1:
        qb = max(1, qb // 2)
        base = b * qb * hq * d * (c.operand_bytes + c.accumulator_bytes)
        per_k = b * (2 * hkv * d * c.operand_bytes + qb * hq * c.accumulator_bytes)
        kb = min(kv, (c.sram_bytes - c.staging_bytes - base) // per_k)
    if kb < 1:
        raise ValueError("one fused attention tile exceeds SRAM")
    totals = Counter()
    qparts = [(qb, q // qb)] + ([(q % qb, 1)] if q % qb else [])
    kparts = [(kb, kv // kb)] + ([(kv % kb, 1)] if kv % kb else [])
    for qr, qcount in qparts:
        for kr, kcount in kparts:
            reps = qcount * kcount
            for m, n, k in ((qr * groups, kr, d), (qr * groups, d, kr)):
                tile = None
                if fixed_variant:
                    ref = gemm_cost(m, n, k, b * hkv, fixed_variant, c)
                    tile = (ref.kt, ref.nt)
                r = gemm_cost(m, n, k, b * hkv, variant, c, tile)
                assert r.hbm_reload_bytes == 0, "fused attention tile must retain its operands"
                for key in ("array_cycles", "useful_macs", "array_energy_J", "sram_energy_J",
                            "vector_energy_J", "sram_read_bytes", "sram_write_bytes",
                            "skew_extra_cycles", "seam_tail_cycles"):
                    totals[key] += getattr(r, key) * reps
                totals["compute_seconds"] += r.seconds * reps
            softmax = 4 * b * hq * qr * kr  # source: NeuSim llm_ops_lib.get_flops_unary_op.
            totals["vector_ops"] += softmax * reps
    blocks = ceil(kv / kb)
    # Derivation: rescale old output, add new output, update max/sum per KV block.
    totals["vector_ops"] += max(0, blocks - 1) * b * hq * q * (2 * d + 4)
    totals["compute_seconds"] += totals["vector_ops"] / c.vector_ops_per_cycle / c.frequency_hz
    totals["vector_energy_J"] += totals["vector_ops"] * c.vector_pj_per_op * 1e-12
    scores = b * hq * q * kv
    score_read = scores * c.accumulator_bytes
    score_write = scores * c.operand_bytes
    totals["sram_read_bytes"] += score_read
    totals["sram_write_bytes"] += score_write
    totals["sram_energy_J"] += (score_read / c.operand_bytes * c.sram_read_pj_per_operand
                                + score_write / c.accumulator_bytes * c.sram_write_pj_per_accumulator) * 1e-12
    totals["peak_live_bytes"] = base + per_k * kb + c.staging_bytes
    totals["hbm_reload_bytes"] = max(0, ceil(q / qb) - 1) * b * kv * hkv * d * 2 * c.operand_bytes
    totals["q_block"] = qb
    totals["kv_block"] = kb
    return dict(totals)


def analyze_operators(ops, chip, analyze_energy=True):
    c = TesseraConfig(**chip.tessera_parameters)
    variant = chip.tessera_variant
    specs = [_native_spec(op, chip) for op in ops]
    remaining = Counter(name for s in specs for name in s.get("inputs", {}))
    store = TensorStore(c.sram_bytes - c.staging_bytes)
    for op, spec in zip(ops, specs):
        inputs, outputs = spec.get("inputs", {}), spec.get("outputs", {})
        writes = spec.get("persistent_write_bytes", 0)
        details = dict(kind=spec["kind"], variant=variant,
                       accounting="analytical", identity=spec.get("identity", "explicit"))
        fixed_variant = spec.get("fixed_mapping_from")
        # Codex: decision start — reserve workspace before deciding HBM residency.
        # Whole live tensors are additional to workspace. This conservative policy
        # may spill an overlapping input, but cannot hide over-capacity residency.
        if spec["kind"] == "gemm":
            m, n, k, b = (spec[t] for t in ("M", "N", "K", "batch"))
            memory = plan_memory(m, n, k, c, b)
            writes += store.reserve(c.sram_bytes - memory.peak_live_bytes)
        elif spec["kind"] == "attention":
            details.update(attention_cost(spec, c, variant, fixed_variant))
            writes += store.reserve(c.sram_bytes - details["peak_live_bytes"])
        reads = sum(size for name, size in inputs.items() if not store.has(name, size))
        for name in inputs:
            remaining[name] -= 1
            if not remaining[name]:
                store.items.pop(name, None)
        for name, size in outputs.items():
            if spec.get("final"):
                writes += size
            elif remaining[name]:
                writes += store.put(name, size, True)
        # Codex: decision end
        if spec["kind"] == "gemm":
            tile = None
            if fixed_variant:
                ref = gemm_cost(m, n, k, b, fixed_variant, c, transfer_bytes=reads, write_bytes=writes)
                tile = (ref.kt, ref.nt)
            r = gemm_cost(m, n, k, b, variant, c, tile, transfer_bytes=reads, write_bytes=writes)
            details.update(asdict(r), M=m, N=n, K=k, batch=b)
            compute = (r.array_cycles + r.reduction_ops / c.vector_ops_per_cycle) / c.frequency_hz
            reads += r.hbm_reload_bytes
        elif spec["kind"] == "attention":
            reads += details["hbm_reload_bytes"]
            compute = details["compute_seconds"]
        else:
            work = spec.get("vector_ops", 0)
            compute = work / c.vector_ops_per_cycle / c.frequency_hz
            sr = sum(inputs.values())
            sw = sum(outputs.values())
            details.update(vector_ops=work, useful_macs=0, array_cycles=0,
                           array_energy_J=0.0, vector_energy_J=work * c.vector_pj_per_op * 1e-12,
                           sram_read_bytes=sr, sram_write_bytes=sw,
                           sram_energy_J=(sr / c.operand_bytes * c.sram_read_pj_per_operand
                                          + sw / c.accumulator_bytes * c.sram_write_pj_per_accumulator) * 1e-12)
        # Initial/final transfers cannot disappear under a max-overlap assumption.
        transfer = (reads + writes) / c.hbm_bytes_per_second
        seconds = overlapped_seconds(compute, reads, writes, c)
        ici_bytes = spec.get("ici_bytes", 0)
        ici_time = ici_bytes / c.ici_bytes_per_second + (c.ici_latency_ns * 1e-9 if ici_bytes else 0)
        seconds += ici_time
        # Each serial operator owns its wall time; background is counted once.
        details.update(seconds=seconds, compute_seconds=compute, hbm_read_bytes=reads,
                       hbm_write_bytes=writes, ici_bytes=ici_bytes,
                       hbm_energy_J=(reads * c.hbm_read_pj_per_byte + writes * c.hbm_write_pj_per_byte) * 1e-12,
                       ici_energy_J=ici_bytes * c.ici_pj_per_byte * 1e-12,
                       background_energy_J=seconds * c.background_power_W)
        details["energy_J"] = sum(details.get(k, 0.0) for k in (
            "array_energy_J", "sram_energy_J", "vector_energy_J", "hbm_energy_J",
            "ici_energy_J", "background_energy_J"))
        op.stats.execution_time_ns = ceil(seconds * 1e9)
        op.stats.sa_time_ns = ceil(details.get("array_cycles", 0) / c.frequency_hz * 1e9)
        op.stats.vu_time_ns = max(0, ceil(compute * 1e9) - op.stats.sa_time_ns)
        op.stats.memory_time_ns = ceil(transfer * 1e9)
        op.stats.ici_time_ns = ceil(ici_time * 1e9)
        op.stats.memory_traffic_bytes = reads + writes
        op.stats.flop_count = 2 * details.get("useful_macs", 0) + spec.get("vector_ops", details.get("vector_ops", 0))
        op.stats.bounded_by = "Memory" if transfer > compute else "Compute"
        op.stats.tessera_details = details
        if analyze_energy:
            op.stats.dynamic_energy_sa_J = details.get("array_energy_J", 0)
            op.stats.dynamic_energy_sram_J = details.get("sram_energy_J", 0)
            op.stats.dynamic_energy_vu_J = details.get("vector_energy_J", 0)
            op.stats.dynamic_energy_hbm_J = details["hbm_energy_J"]
            op.stats.dynamic_energy_ici_J = details["ici_energy_J"]
            op.stats.static_energy_other_J = details["background_energy_J"]
        assert store.size <= store.capacity
    return ops
