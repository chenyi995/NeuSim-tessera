"""Explicit inference graphs for the NeuSim Tessera backend.

Model dimensions and batch compositions are supplied from saved workload
metadata. Tensor names encode true dependencies; independent requests never
share KV tensors. All operators run through NeuSim's public analysis entry.
"""
from __future__ import annotations

from math import prod

from neusim.npusim.frontend.Operator import Operator, OpType, OpcodeType, Tensor


def make_op(name, spec, count=1):
    kind = spec["kind"]
    op = Operator(name=name, description=name, opcode=name, tessera_spec=spec)
    op.stats.count = count
    op.op_type = OpType.MXU if kind in ("gemm", "attention") else OpType.VPU
    op.opcode_type = {"gemm": OpcodeType.EINSUM, "attention": OpcodeType.FLASH_ATTENTION}.get(kind, OpcodeType.ELEMENTWISE)
    # Shape detail is in tessera_spec; the tensor byte counts remain explicit.
    op.input_tensors = [Tensor.from_shape(key, [size], "UINT8") for key, size in spec.get("inputs", {}).items()]
    op.output_tensors = [Tensor.from_shape(key, [size], "UINT8") for key, size in spec.get("outputs", {}).items()]
    return op


def llm_graph(model, requests, operand_bytes, *, layers=None, fixed_mapping=False,
              include_boundary=True, compact_layers=False):
    """One complete model step for (query length, current KV length) requests.

    A prefill is q>1; decode is q=1. The caller advances KV lengths each step.
    All layer operations are serialized; Phi's parallel residual branches are
    conservatively serialized on shared compute resources.
    """
    dmodel = model["embedding_dim"]
    tp = model["tensor_parallel_size"]
    heads = model["num_q_heads"] // tp
    kvheads = max(1, model["num_kv_heads"] // tp)
    dim = model["embedding_dim"] // model["num_q_heads"]
    ffn = model["mlp_hidden_dim"] // tp
    layers = model["num_layers"] if layers is None else layers
    gated = model["use_gated_mlp"]
    phi = not model["post_attn_norm"]
    total_tokens = sum(q for q, _ in requests)
    if total_tokens < 1 or any(q < 1 or length < q for q, length in requests):
        raise ValueError("invalid request lengths")
    widthbytes = total_tokens * dmodel * operand_bytes
    ops = []

    def emit(name, kind="vector", inputs=None, outputs=None, **kw):
        spec = dict(kind=kind, inputs=inputs or {}, outputs=outputs or {}, **kw)
        if fixed_mapping and kind in ("gemm", "attention"):
            spec["fixed_mapping_from"] = "tessera"
        # Codex: decision start — saved Phi metadata enables explicit projection bias additions.
        bias = kind == "gemm" and model["use_qkv_bias" if name.endswith("qkv_projection") else "use_bias"]
        if bias:
            target = spec["outputs"]
            spec["outputs"] = {key + "_unbiased": size for key, size in target.items()}
            ops.append(make_op(name, spec))
            bias_inputs = dict(spec["outputs"])
            bias_inputs[name + "_bias_weight"] = kw["N"] * operand_bytes
            ops.append(make_op(name + "_bias", dict(kind="vector", inputs=bias_inputs,
                outputs=target, vector_ops=kw["M"] * kw["N"])))
        else:
            ops.append(make_op(name, spec))
        # Codex: decision end

    x = "hidden_input"
    if include_boundary:
        emit("embedding", inputs={"embedding_rows": widthbytes}, outputs={x: widthbytes}, vector_ops=0)
    body_start = len(ops)
    for layer in range(1 if compact_layers else layers):
        p = f"layer{layer}/"
        normalized = p + "norm"
        # Source: NeuSim npusim_lib elementwise model, LayerNorm=8 / RMSNorm=6.
        norm_ops = 8 if phi else 6
        norm_params = dmodel * operand_bytes * (2 if phi else 1)
        emit(p + "input_norm", inputs={x: widthbytes, p + "norm_weight": norm_params},
             outputs={normalized: widthbytes}, vector_ops=norm_ops * total_tokens * dmodel)
        qkv_outputs = {}
        for i, (q, _) in enumerate(requests):
            for label, h in (("q", heads), ("k", kvheads), ("v", kvheads)):
                qkv_outputs[p + f"{label}{i}"] = q * h * dim * operand_bytes
        qkv_width = (heads + 2 * kvheads) * dim
        emit(p + "qkv_projection", "gemm", {normalized: widthbytes, p + "wqkv": dmodel * qkv_width * operand_bytes},
             qkv_outputs, M=total_tokens, N=qkv_width, K=dmodel, batch=1)
        att_outputs = {}
        for i, (q, length) in enumerate(requests):
            qi, ki, vi = (p + f"{t}{i}" for t in ("q", "k", "v"))
            qr, kr = qi + "_rope", ki + "_rope"
            # RoPE: two multiplies and one add per rotated element; sin/cos precomputed.
            rotated = int(q * (heads + kvheads) * dim * model["partial_rotary_factor"])
            emit(p + f"rope{i}", inputs={qi: qkv_outputs[qi], ki: qkv_outputs[ki]},
                 outputs={qr: qkv_outputs[qi], kr: qkv_outputs[ki]}, vector_ops=3 * rotated)
            emit(p + f"kv_append{i}", inputs={kr: qkv_outputs[ki], vi: qkv_outputs[vi]},
                 persistent_write_bytes=qkv_outputs[ki] + qkv_outputs[vi], vector_ops=0)
            out = p + f"attention{i}"
            size = q * heads * dim * operand_bytes
            emit(out, "attention", {qr: qkv_outputs[qi], p + f"cache_k{i}": length * kvheads * dim * operand_bytes,
                                     p + f"cache_v{i}": length * kvheads * dim * operand_bytes},
                 {out: size}, q=q, kv=length, hq=heads, hkv=kvheads, d=dim, batch=1)
            att_outputs[out] = size
        joined = p + "attention_join"
        emit(joined, inputs=att_outputs, outputs={joined: sum(att_outputs.values())}, vector_ops=0)
        projected = p + "projected"
        emit(p + "out_projection", "gemm", {joined: sum(att_outputs.values()), p + "wo": heads * dim * dmodel * operand_bytes},
             {projected: widthbytes}, M=total_tokens, N=dmodel, K=heads * dim, batch=1)
        if tp > 1:
            # Ring allreduce sends 2*(tp-1)/tp tensor bytes per rank.
            reduced = projected + "_reduced"
            emit(p + "attention_allreduce", inputs={projected: widthbytes}, outputs={reduced: widthbytes},
                 vector_ops=total_tokens * dmodel * (tp - 1) / tp,
                 ici_bytes=ceil_bytes(2 * (tp - 1) * widthbytes, tp))
            projected = reduced
        if phi:
            ffin = normalized
        else:
            residual = p + "residual"
            emit(p + "attention_residual", inputs={x: widthbytes, projected: widthbytes},
                 outputs={residual: widthbytes}, vector_ops=total_tokens * dmodel)
            ffin = p + "ffn_norm"
            emit(ffin, inputs={residual: widthbytes, p + "ffn_norm_weight": dmodel * operand_bytes},
                 outputs={ffin: widthbytes}, vector_ops=6 * total_tokens * dmodel)
        up = p + "up"
        expanded = ffn * (2 if gated else 1)
        upbytes = total_tokens * expanded * operand_bytes
        emit(p + "ffn_up", "gemm", {ffin: widthbytes, p + "wup": dmodel * expanded * operand_bytes},
             {up: upbytes}, M=total_tokens, N=expanded, K=dmodel, batch=1)
        act = p + "act"
        actbytes = total_tokens * ffn * operand_bytes
        # Scalar operation accounting: SiLU x*sigmoid(x) and gate*up = 5;
        # tanh-approx GELU polynomial, tanh and final multiplies = 8.
        emit(p + "silu_gate" if gated else p + "gelu", inputs={up: upbytes},
             outputs={act: actbytes}, vector_ops=(5 if gated else 8) * total_tokens * ffn)
        down = p + "down"
        emit(p + "ffn_down", "gemm", {act: actbytes, p + "wdown": ffn * dmodel * operand_bytes},
             {down: widthbytes}, M=total_tokens, N=dmodel, K=ffn, batch=1)
        if tp > 1:
            reduced = down + "_reduced"
            emit(p + "ffn_allreduce", inputs={down: widthbytes}, outputs={reduced: widthbytes},
                 vector_ops=total_tokens * dmodel * (tp - 1) / tp,
                 ici_bytes=ceil_bytes(2 * (tp - 1) * widthbytes, tp))
            down = reduced
        next_x = p + "output"
        residual_inputs = {x: widthbytes, projected: widthbytes, down: widthbytes} if phi else {residual: widthbytes, down: widthbytes}
        emit(p + "ffn_residual", inputs=residual_inputs, outputs={next_x: widthbytes},
             vector_ops=(len(residual_inputs) - 1) * total_tokens * dmodel)
        x = next_x
    if compact_layers:
        # Codex: decision start — repeated homogeneous blocks have distinct weights but identical costs.
        for op in ops[body_start:]:
            op.stats.count = layers
        # Codex: decision end
    if include_boundary:
        # One logit row per request per replay step, including prefill chunks.
        last = "last_token_hidden"
        lastbytes = len(requests) * dmodel * operand_bytes
        emit("last_token_gather", inputs={x: widthbytes}, outputs={last: lastbytes}, vector_ops=0)
        normed = "final_normed"
        emit("final_norm", inputs={last: lastbytes, "final_norm_weight": dmodel * operand_bytes * (2 if phi else 1)},
             outputs={normed: lastbytes}, vector_ops=(8 if phi else 6) * len(requests) * dmodel)
        vocab = ceil_bytes(model["vocab_size"], tp)
        logitsbytes = len(requests) * vocab * operand_bytes
        emit("lm_head", "gemm", {normed: lastbytes, "lm_head_weight": dmodel * vocab * operand_bytes},
             {"logits": logitsbytes}, M=len(requests), N=vocab, K=dmodel, batch=1)
        emit("argmax", inputs={"logits": logitsbytes}, outputs={"token_ids": len(requests) * 4},
             vector_ops=len(requests) * vocab, final=True)
    elif ops:
        ops[-1].tessera_spec["final"] = True
    return ops


def ceil_bytes(a, b):
    return (a + b - 1) // b


def native_llm_graph(model, requests, config, *, layers=None, compact_layers=True):
    """Lower the explicit full inference graph to native NeuSim operators.

    GEMM and fused attention retain semantic shapes. Other fused kernels supply
    the graph's documented scalar work and exact input/output tensor bytes to
    the native VU throughput and tensor-size memory models. Native operator
    residency is retained (operators do not inherit the custom backend's LRU).
    """
    from neusim.npusim.frontend import llm_ops_lib as lib
    from math import ceil
    source = llm_graph(model, requests, 2, layers=layers, compact_layers=compact_layers)
    result = []
    for original in source:
        spec = original.tessera_spec
        name = original.name.replace("/", "_")
        count = original.stats.count
        if spec["kind"] == "gemm":
            m, n, k = (spec[key] for key in ("M", "N", "K"))
            op = lib.create_einsum_op([m, k], [k, n], "MK;KN->MN", name=name, count=count)
        elif spec["kind"] == "attention":
            b, q, kv, hq, hkv, d = (spec[key] for key in ("batch", "q", "kv", "hq", "hkv", "d"))
            op = lib.create_multi_head_flash_attention_op([b, q, hq, d], [b, kv, hkv, d], [b, kv, hkv, d], name=name, count=count)
        elif spec.get("ici_bytes", 0):
            # Native collective factory determines communication and VU counts.
            elements = next(iter(spec["outputs"].values())) // 2
            op = lib.create_all_reduce_op([elements], [model["tensor_parallel_size"]],
                                          [config.ici_bw_GBps], config, name=name, count=count)
        else:
            # Codex: decision start — fused scalar kernels retain their explicit
            # graph work; byte-shaped tensors provide exact native memory traffic.
            inputs = [Tensor.from_shape(key, [value], "DT_UINT8") for key, value in spec["inputs"].items()]
            output_bytes = sum(spec["outputs"].values()) + spec.get("persistent_write_bytes", 0)
            outputs = [Tensor.from_shape(name + "_out", [output_bytes], "DT_UINT8")]
            work = ceil(spec.get("vector_ops", 0))
            op = Operator(name=name, description=name, opcode="TesseraVector", config_str="TesseraVector()",
                          op_type=OpType.VPU, opcode_type=OpcodeType.ELEMENTWISE,
                          input_tensors=inputs, output_tensors=outputs,
                          input_tensor_shape_str=lib.format_input_tensor_shapes([t.shape for t in inputs], "DT_UINT8"),
                          output_tensor_shape_str=lib.format_output_tensor_shape([output_bytes], "DT_UINT8"),
                          tessera_spec=dict(kind="native_vector", vector_ops=work))
            op.stats.flop_count = work
            op.stats.count = count
            # Codex: decision end
        result.append(op)
    return result
