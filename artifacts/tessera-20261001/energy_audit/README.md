# Independent WS8 energy audit

This is a read-only analysis of completed simulations. No simulator code, coefficient, mapping or run result was changed.

## Native SRAM accounting

Both reads and writes contribute transferred bytes, including padded operand reads, partial writes and reduction reads/writes. The native model uses one aggregate dynamic coefficient for SRAM transfers; it does not distinguish read from write energy. The ideal per-PE bandwidth affects service time. Energy retains the native reference bytes/ns coefficient.

| Architecture | SRAM pJ/byte | SRAM pJ/bit | SRAM static W | Chip static W | Compute pJ/op |
|---|---:|---:|---:|---:|---:|
| WS | 2.04 | 0.26 | 24.21 | 105.03 | 6.302 |
| WS-independent-8 | 2.04 | 0.26 | 24.21 | 105.03 | 6.302 |
| Tessera-8 | 2.04 | 0.26 | 24.21 | 105.03 | 6.303 |

These coefficients precede regulator losses. Whole-chip static power is shared across architectures under the revision model and accumulates over execution and recorded idle time. The array-plus-SRAM area estimate affects the horizontal plot axis, not energy efficiency.

## Llama-2-7B chat: complete E2E energy

| Tiling | Architecture | Delay (s) | SRAM traffic (TB) | Compute dynamic (kJ) | SRAM dynamic (kJ) | HBM dynamic (kJ) | All static (kJ) | Total (kJ) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| native_tail | WS | 43711.56 | 25815.38 | 2198.89 | 62.02 | 1.60 | 5139.60 | 7405.59 |
| native_tail | WS-independent-8 | 440.47 | 6727.51 | 39.42 | 15.95 | 2.65 | 51.59 | 110.62 |
| native_tail | Tessera-8 | 1459.75 | 20105.14 | 158.45 | 47.76 | 1.63 | 171.29 | 382.84 |
| joint_edp | WS | 739.95 | 552.51 | 33.40 | 1.33 | 1.66 | 86.98 | 123.43 |
| joint_edp | WS-independent-8 | 182.56 | 5602.67 | 33.11 | 13.24 | 2.67 | 21.32 | 71.19 |
| joint_edp | Tessera-8 | 174.09 | 722.87 | 33.10 | 1.74 | 1.57 | 20.47 | 56.95 |

All-static already includes SRAM static energy. Total also includes VU/ICI/other dynamic energy, retained in the CSV.

## Independent WS8 versus Tessera-8 for every workload

| Tiling | Workload | WS8/T8 SRAM traffic | WS8/T8 total energy | WS8 speedup |
|---|---|---:|---:|---:|
| native_tail | Llama-2-7B chat | 0.33× | 0.29× | 3.31× |
| native_tail | Llama-3-8B chat | 0.85× | 0.80× | 0.91× |
| native_tail | Llama-2-70B chat | 0.87× | 5.06× | 0.08× |
| native_tail | Phi-2 chat | 1.59× | 0.85× | 1.24× |
| native_tail | Llama-2-7B code | 0.43× | 0.40× | 2.04× |
| native_tail | Llama-2-7B arXiv | 0.44× | 0.39× | 2.42× |
| native_tail | FlashInfer GQA | 3.90× | 0.65× | 1.87× |
| native_tail | FlashInfer MQA | 7.17× | 1.08× | 0.99× |
| native_tail | CacheBlend SAMSum | 0.34× | 5.05× | 0.10× |
| native_tail | CacheBlend WikiMQA | 0.93× | 15.07× | 0.04× |
| native_tail | EPIC HotpotQA | 0.50× | 17.70× | 0.03× |
| native_tail | EPIC Multi-News | 0.29× | 2.09× | 0.19× |
| joint_edp | Llama-2-7B chat | 7.75× | 1.25× | 0.95× |
| joint_edp | Llama-3-8B chat | 7.97× | 1.25× | 0.96× |
| joint_edp | Llama-2-70B chat | 11.33× | 1.27× | 0.97× |
| joint_edp | Phi-2 chat | 5.40× | 1.22× | 0.93× |
| joint_edp | Llama-2-7B code | 11.54× | 1.22× | 0.99× |
| joint_edp | Llama-2-7B arXiv | 7.08× | 1.25× | 0.94× |
| joint_edp | FlashInfer GQA | 2.52× | 1.02× | 1.00× |
| joint_edp | FlashInfer MQA | 2.72× | 1.02× | 1.00× |
| joint_edp | CacheBlend SAMSum | 13.56× | 1.28× | 0.98× |
| joint_edp | CacheBlend WikiMQA | 14.10× | 1.28× | 0.97× |
| joint_edp | EPIC HotpotQA | 13.79× | 1.29× | 0.95× |
| joint_edp | EPIC Multi-News | 12.14× | 1.27× | 0.98× |

## Scope of the conclusion

The figures compare total E2E energy efficiency, rather than SRAM efficiency or SRAM power alone. The native small-K-tile issue identified in the preceding diagnostic remains present. The normal policy restricts connected-array fragmentation to residual strips/corners, while independent WS always uses its fixed small arrays. The EDP policy searches array geometries and SRAM tiles jointly.

The SRAM coefficient is NeuSim's aggregate reference power divided by reference bandwidth. It is not a per-bank macro characterization for each architecture, and it does not add architecture-specific SRAM bank/periphery or distribution-network power. The requested ideal SRAM bandwidth remains in force. This audit establishes consistency with that model, not post-layout physical power accuracy.

Source code: `power_model.analyze_dynamic_energy`, `ChipConfig.vmem_bw_GBps`, `tessera_baselines.counts`, and `replay_tessera_arrivals.Costs.integrate`.

All source numbers are read and transformed by this retained script. Per-operator SRAM charges were independently reconstructed from recorded traffic and regulator efficiency, and output CSVs were read back.
