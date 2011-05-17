from .base import *

import operator

__all__ = ['AddNode', 'SubNode', 'MulNode', 'DivNode']

AddNode = make_trivial(operator.add)
SubNode = make_trivial(operator.sub)
MulNode = make_trivial(operator.mul)
DivNode = make_trivial(operator.truediv)
