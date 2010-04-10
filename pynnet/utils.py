r"""
Module for useful hacks that people have found to get things done.

If ever a proper abstraction for the things done here is found, then
the functionality will be moved.  But in the meantime, do use this.
"""
from base import *
from layers.base import BaseLayer, CompositeLayer

__all__ = ['walk_layers']

def walk_layers(f, net, dtype=BaseLayer):
    r"""
    Walks the tree of layers starting at `net`, calling `f` on each
    layer of type `dtype` encoutered.

    :notests:
    """
    if isinstance(net, dtype):
        f(net)
    if isinstance(net, CompositeLayer):
        for l in net._dict.values():
            walk_layers(f, l, dtype)
