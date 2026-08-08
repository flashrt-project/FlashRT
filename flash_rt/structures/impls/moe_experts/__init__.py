"""``moe_experts`` structure family.

The structure is the expert bank of a sparse-MoE block: one module
holding every expert's projection weights as stacked 3D tensors, called
with the token batch plus the router's top-k assignment. On the hosts
this family serves, that bank is where nearly all of the checkpoint's
weight mass lives — which is exactly why it is the seam worth owning
when the dense checkpoint does not fit the card.
"""
