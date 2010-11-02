r"""
Module for useful hacks that people have found to get things done.

If ever a proper abstraction for the things done here is found, then
the functionality will be moved.  But in the meantime, do use this.
"""
from base import *
from pynnet.nodes.base import BaseNode

__all__ = ['walk_nodes']

def walk_nodes(f, net, dtype=BaseNode):
    r"""
    Walks the tree of nodes starting at `net`, calling `f` on each
    node of type `dtype` encoutered.

    :notests:
    """
    if isinstance(net, dtype):
        f(net)
    for n in net.inputs:
        walk_nodes(f, n, dtype)
