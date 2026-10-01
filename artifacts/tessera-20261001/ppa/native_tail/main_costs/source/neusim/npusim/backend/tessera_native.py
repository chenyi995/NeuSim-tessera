"""A deliberately narrow SA-cycle replay hook for native NeuSim comparisons.

This imports array timing only. Native VU selection, accumulation, tiling,
memory bandwidth, overlap, utilization and component power remain responsible
for the operator result. It is not a general Tessera compiler or traffic model.
"""
import math


def replay_sa_time(node_cost, config, b, m, n, k):
    # Codex: decision start — reject mismatched replay inputs rather than silently
    # attributing another shape's cycles to this native operator.
    spec = node_cost.tessera_spec
    if not isinstance(spec, dict) or spec.get("kind") != "native_sa_replay":
        raise ValueError("native SA replay requires an explicit native_sa_replay spec")
    if tuple(spec.get(key) for key in ("B", "M", "N", "K")) != (b, m, n, k):
        raise ValueError("native SA replay shape does not match the einsum")
    if spec.get("pe_count") != config.num_sa * config.sa_dim ** 2:
        raise ValueError("native SA replay PE budget does not match ChipConfig")
    cycles = spec.get("sa_cycles")
    if type(cycles) is not int or cycles < 1 or config.freq_GHz <= 0:
        raise ValueError("native SA replay requires positive integer cycles and frequency")
    if cycles * spec["pe_count"] < b * m * n * k:
        raise ValueError("native SA replay violates the useful-MAC throughput bound")
    node_cost.stats.tessera_details = dict(spec)
    return math.ceil(cycles / config.freq_GHz)
    # Codex: decision end
