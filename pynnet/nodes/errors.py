from pynnet.nodes.base import *
from pynnet import errors

# This will error out when a new function is added to errors.
# (which is what we want).
__all__ = errors.__all__

nll = make_trivial(errors.nll)
mse = make_trivial(errors.mse)
class_error = make_trivial(errors.class_error)
cross_entropy = make_trivial(errors.cross_entropy)
