# Array timing versus the Tessera paper

The measured boundary below is the first weight beat entering the array,
through the last result leaving it, inclusive. It excludes transpose fill,
Gemmini input staging, scratchpad traffic, DMA, and output writeback.

The source formulas are `sec/03_ttsa.tex:206` and
`sec/04_design.tex:284` in the Tessera paper checkout.

| RTL test | M | R | Measured fold latency | Measured II | Paper fold latency |
|---|---:|---:|---:|---:|---:|
| 8x8 square | 19 | 3 | 34 | 19 | 34 |
| 8x8 square | 4 | 3 | 19 | 8 | 19 |
| 32x32 square, g=32 | 32 | 3 | 95 | 32 | 95 |
| 32x32 square over four width-8 strips | 32 | 3 | 98 | 32 | 95 |
| Two 8x8 squares, wide mirror chain | 19 | 3 | 43 | 19 | 42 |
| Two 8x8 squares, tall mirror chain | 19 | 3 | 43 | 19 | 42 |

All outputs are checked against independent matrix multiplication, with
distinct consecutive weights and concurrent preload/compute/output.
Reproduce with `python3 partition/test_rtl.py tessera 8`,
`python3 partition/test_rtl.py tessera 8 --rows 4`, and
`python3 partition/test_rtl.py mirror 8` after elaborating those modules.

The square implements `L = M + 2D - 1` and `II = max(M,D)`.
The mirror pair adds one boundary register, matching the original SA-rtl
strip-to-strip connection. Its measured latency is `H + M + W - 1 + 1`.
The transpose fills for another H cycles before this measured load boundary;
for H=8, M=19, W=16, the first ingress beat through last output takes 51
cycles (43+8). Subsequent fills overlap ongoing work in independent banks.

There is a separate small-M ambiguity in the paper's R-round formula.
If completion means the last valid result, the exact no-bubble schedule is
`H + (R-1)*max(M,H) + M + W - 1` (plus registered seam delays).
The printed `H + R*max(M,H) + W - 1` agrees when M>=H, but includes
H-M idle cycles after the last fold when M<H. The 8x8, M=4, R=3 simulation
finishes at cycle 35 inclusive; the printed round formula gives 39.
This observation does not modify the paper or claim a system-level speedup.

The current full Gemmini adapter additionally stages operands, waits for
all outputs of a compute session, and drains them to the accumulator before
starting the next session. Its end-to-end II is not yet the paper's array II.
Double buffering is functional: a compute/preload instruction pair consumes
one stationary bank while loading the other, with assertions against bank
overwrites and independent weight/psum transport. This alone does not imply
that DMA and all control overhead are hidden.

## Physical 32x32 merged fabric

With M=32 and three consecutive folds, all 3072 numerical outputs pass:

| Minimum side g | First-fold cycles | II | Three-fold completion |
|---|---:|---:|---:|
| 32 | 95 | 32 | 159 |
| 16 | 96 | 32 | 160 |
| 8 | 98 | 32 | 162 |
| 4 | 102 | 32 | 166 |
| 2 | 110 | 32 | 174 |

The physical seam contribution is 32/g-1, so this implementation gives
L=M+2D-1+(D/g-1) for the merged D=32 square. It increases the final tail;
it does not increase the steady-state II. g=32 is separately generated
hardware; switching g=8 to merged mode retains its three seams and its area.

## Full Gemmini system, corrected adapter v3

All backends execute the same three 32x32x32 GEMMs, with 3072 bit-exact FP32
comparisons, 24576 bytes read and 12288 bytes written. Latency is the first
main-memory request through the final result-write acknowledgement, inclusive.
The C++ harness has 8-cycle memory response latency and a 128-bit data bus.

| g=8 backend | Double buffer | Single buffer | Cycles saved by DB |
|---|---:|---:|---:|
| Tessera | 2447 | 2518 | 71 |
| plaA | 2550 | 2577 | 27 |
| plaB | 2550 | 2577 | 27 |

The streamed refill test preloads two operand sets, queues enough execute
commands for Gemmini's three-command RAW-hazard lookahead, and admits the
third main-memory refill after the first PE begins computing. Double-buffer
runs have PE load/compute overlap of 1024 PE-cycles (Tessera) and 224 (Planaria)
and DMA/compute overlap of 119 and 242 cycles respectively. Single-buffer
runs have zero PE load/compute overlap, while DMA still overlaps computation.
Different PE overlap counts reflect a one-cycle Tessera lock versus shifting
Planaria stationary weights through the loading bank; they are not utilization.

Compared with the earlier prototype, v3 uses column-banked result writes and
releases ingress storage at the actual final weight beat, eliminating a
conservative preload-only settling wait. Earlier 2550/2601 g=8 DB numbers
are superseded. Cross-grain v3 results are in `results/system_latency.csv`.

The full-system adapter still stages and drains complete sessions. Its latency
and II must not be substituted for, or claimed equal to, the paper's isolated
array formulas. The separate pure-fabric tests establish the array schedule.
