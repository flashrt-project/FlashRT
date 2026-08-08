"""Beta: the join between two bound structures, declared rather than implied.

Every structure in the catalog declares its own boundary. What none of
them declares is the *join* — what has to agree between a structure and
the one it feeds. Today those agreements exist only inside each impl's
code, so they are never checked, never negotiated, and never optimised
across. Every join therefore defaults to "materialise a fresh tensor in
the host's convention", which is what a hand-written runtime never does:
there the author holds the whole dataflow and picks, per join, the same
buffer, the same layout, no re-quantisation.

The vocabulary here is derived, not designed. Each attribute exists
because one specific join cost real time or gave a wrong verdict, and
each carries that incident in its docstring. The evidence for the whole
idea is one controlled comparison: ``dtype`` is the only join attribute
that was ever declared and negotiated, and it is the only one that
stopped causing failures. The other five were never declared and every
one of them bit.

**This is beta.** Three rules keep it honest, and they double as the
conditions for deleting it:

1. *Descriptive before prescriptive.* It must correctly describe the
   joins that already work before it is allowed to change any behaviour.
   A vocabulary that cannot express what the stack already does is the
   wrong vocabulary.
2. *No attribute without a negotiator.* An attribute that no join
   negotiates and no measurement depends on gets deleted, not
   documented.
3. *Off by default.* Nothing in the main path consumes this unless asked
   to, and an attribute a port does not declare is simply not negotiated
   — so coverage can only grow.

The deprecation signal is stated up front: if each new structure needs a
new attribute, this is not a vocabulary, it is a junk drawer. The
attribute set is supposed to converge as structures are added. If it
does not, delete this package.
"""

from .negotiate import JoinRefused, negotiate
from .ports import ATTRIBUTES, Join, Port

__all__ = ["ATTRIBUTES", "Join", "JoinRefused", "Port", "negotiate"]
