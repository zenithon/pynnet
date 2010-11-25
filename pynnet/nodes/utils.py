from pynnet.nodes.base import *

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
    Node to select a portion of the input.

    See the `split` function for a user-friendly version.
    
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
        Tests:
        >>> x = T.fmatrix('x')
        >>> s = SplitNode(x, i[1:,:])
        >>> s.split
        (slice(1, None, None), slice(None, None, None))
        >>> s.params
        []
        """
        BaseNode.__init__(self, [input], name)
        self.split = split

    def transform(self, inp):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> s = SplitNode(x, i[1:3])
        >>> theano.pp(s.output)
        'x[1:3]'
        """
        return inp[self.split]

def split(input, splits, name=None):
        r"""
        Splits an input into one or more sub-regions.

        The sub-region can overlap and do not have to be in any
        particular order.

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
    r"""
    Node to join one or more nodes together.

    Input nodes must have the same dimensions apart from the first one.
    """
    def __init__(self, inputs, name=None):
        r"""
        :nodoc:
        """
        BaseNode.__init__(self, inputs, name)

    def transform(self, *inputs):
        r"""
        Tests:
        >>> x, y, z = T.fmatrices('xyz')
        >>> j = JoinNode([x, y, z])
        >>> theano.pp(j.output)
        'join(0, x, y, z)'
        """
        return T.join(0, *inputs)
