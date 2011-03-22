from .base import *
from pynnet import errors

# This will error out when a new function is added to errors.
# (which is what we want).
__all__ = errors.__all__

nll = make_trivial(errors.nll)
mse = make_trivial(errors.mse)
class_error = make_trivial(errors.class_error)
binary_cross_entropy = make_trivial(errors.binary_cross_entropy)
multi_cross_entropy = make_trivial(errors.multi_cross_entropy)
