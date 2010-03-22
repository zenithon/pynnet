r"""
Module for useful hacks that people have found to get things done.

If ever a proper abstraction for the things done here is found, then
the functionality will be moved.  But in the meantime, do use this.
"""
from base import *
from pynnet import SharedConvLayer

__all__ = ['walk_layers', 'adjust_imgshape', 'clear_imgshape']

def walk_layers(f, net, dtype=BaseObject):
    if dtype in type(net).mro():
        f(net)
    if hasattr(net, 'layers'):
        for l in net.layers:
            walk_layers(f, l, dtype)
    if hasattr(net, 'layer'):
        walk_layers(f, net.layer, dtype)

def adjust_imgshape(net, bsize):
    def f(l):
        l.image_shape = (bsize, l.image_shape[1],
                         l.image_shape[2], l.image_shape[3])
    walk_layers(f, net, dtype=SharedConvLayer)

def clear_imgshape(net):
    def f(l):
        l.image_shape = None
    walk_layers(f, net, dtype=SharedConvLayer)

