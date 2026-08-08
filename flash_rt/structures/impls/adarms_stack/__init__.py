"""The adaptive-RMS decoder stack region: a cached-prefix expert tower.

An action-expert decoder whose every norm is conditioned (scale, shift,
gate from one dense projection), attending over a prefix another tower
left in the cache. One region family identifies the stack shape; its
fused-chain candidate re-expresses the whole per-layer loop in hub
primitives with static-FP8 GEMMs.
"""
