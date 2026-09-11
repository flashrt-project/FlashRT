"""vla_tick_pipeline: schedule-layer stage structure (cond_iter_pipeline
family, tick specialization).

Unlike region structures there is no standalone torch reference module:
the parity reference is the host's own eager path under the same noise
window (see structure.yaml gates.parity.reference).
"""
