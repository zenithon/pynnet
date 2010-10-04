from base import *

# Since these nodes are more or less patches or fixes for small
# problems we do not export them.  They should only be used explicitly
# in wrapper functions.
__all__ = ['i', 'split', 'JoinNode']

class iClass(object):
    r"""
    Helper class to pass slices to methods.

    Examples:
    >>> i[0:1]
    slice(0, 1, None)
    >>> i[:,3:]
    (slice(None, None, None), slice(3, None, None))
    """
    def __getitem__(self, slice):
        r"""
        Returns the slice argument.
        """
        return slice

i = iClass()

class SplitNode(BaseNode):
    r"""
    Class to split an input expression into one or more pieces.
    
    The regions are specified by a list of tuples representing indexes
    into the input. The split regions may overlap.

    The first split is the output of the main class and the other will
    be the outputs of the subnodes accessible though `helpers`.

    Note: To use the convinient syntax in the examples below import
    `i` from this module along with SplitNode.  Otherwise you can
    pass instances of `slice` with the appropriate parameters.

    Note2: This class doesn't compute its output shape in any case.
    (since I am lazy)  If you want to do it I will be glad.

    Examples:
    >>> x = T.ivector()
    >>> s = SplitNode(x, i[:,0])
    >>> s = SplitNode(x, i[50:150,:])
    >>> s = SplitNode(x, i[2])

    Attributes:
    `split` -- the section of the input this class will return.
    """
    def __init__(self, input, split, name=None):
        r"""
        Initialize fields.
        
        Tests:
        >>> x = T.fmatrix('x')
        >>> s = SplitNode(x, i[1:,:])
        >>> s.split
        (slice(1, None, None), slice(None, None, None))
        >>> s.params
        []
        """
        self.split = split
        BaseNode.__init__(self, [input], name)

    def transform(self, inp):
        r"""
        Build node with input expression `input`.
        
        Tests:
        >>> x = T.fmatrix('x')
        >>> s = SplitNode(x, i[1:3])
        >>> theano.pp(s.output)
        'x[1:3]'
        """
        return inp[self.split]

def split(input, splits, name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> s1, s2 = split(x, [i[0,2], i[1:,:]])
        >>> s1.split
        (0, 2)
        >>> s2.split
        (slice(1, None, None), slice(None, None, None))
        """
        if name is None:
            return tuple(SplitNode(input, s) for s in splits)
        else:
            return tuple(SplitNode(input, s, '%s-%d'%(name, i)) for i, s in enumerate(splits))

class JoinNode(BaseNode):
    def transform(self, *inputs):
        return T.join(0, *inputs)
