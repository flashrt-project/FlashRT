"""The DiT stack region: one span, more than one executable shape.

An alternating self/cross DiT block stack is the first region family —
the span where hardware disagreed about structure itself. On a device
whose hub packages ship the fused NVFP4 epilogue symbols, the fastest
form is a launch chain: norms emit FP4 directly, residuals ride the
GEMM epilogues, the per-step modulators come from a bind-time table.
On a device without those symbols the seat-by-seat composition is the
form, and nothing here activates. The choice is a receipt
(:mod:`flash_rt.structures.regions`), never a device branch.
"""
