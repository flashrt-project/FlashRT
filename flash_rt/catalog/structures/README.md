# Structure catalog

## What this is

A structure entry is a **local boundary expression**: what a computation
region is — its symbolic dimensions, inputs/outputs, weight slots, variant
semantics — plus an executable reference implementation that defines what
"the same computation" means. The catalog exists for two purposes:

1. **Context alignment against native pipelines.** When an adapter or a
   host integration is being built (or read), the catalog is the map that
   says which region of the host graph corresponds to which structure and
   under which variant — so N host implementations of one boundary can be
   compared, ported, and reasoned about as one thing.
2. **Distribution boundary management on the torch side** — the frontend,
   discovery, and swap machinery consume these boundaries to decide what
   can be handed to an implementation and at what seam.

## What this is not

**The catalog adjudicates nothing.** No performance claims, no expected
wins, no negative results, no campaign case histories, no tuning guidance.
Every judgment of that kind is conditional on a model, a hardware
generation, a driver, and a host version, and it expires the moment any of
those move — the only arbiter of whether an implementation is correct or
faster is a test run against the live system, never a statement recorded
here. Dated results belong to campaign records and per-binding
qualification gates, which are re-established by re-running them, not by
being quoted.

Practically, for an entry in this directory:

- **Belongs here**: boundary math, dimension/stride contracts, variant
  *semantics* (including correctness-critical ones such as state snapshot
  or rollback semantics — properties of the computation itself), the
  executable reference, version numbers.
- **Does not belong here**: throughput or latency numbers, "X was judged
  negative/positive", hardware-specific observations, host-specific war
  stories, anything phrased as a verdict. If it can go stale without this
  file changing, it goes elsewhere.
