from base import *

# Since these layers are more or less patches or fixes for small
# problems we do not export them.  They should only be used explicitly
# in wrapper functions.
__all__ = []

class IdentityLayer(BaseLayer):
    r"""
    Class that copies input to output.  Should only be useful when combining
    with layers like OpLayer or JoinLayer when you need the input untouched.

    Examples:
    >>> i = IdentityLayer()
    """
    def __init__(self, name=None):
        r"""
        Tests:
        >>> i = IdentityLayer()
        >>> i2 = test_saveload(i)
        """
        BaseLayer.__init__(self, name)

    def build(self, input, input_shape=None):
        r"""
        Build layer with input expression `input`.
        
        Tests:
        >>> i = IdentityLayer()
        >>> x = T.fmatrix('x')
        >>> i.build(x, (1, 2))
        >>> i.input
        x
        >>> i.output_shape
        (1, 2)
        >>> i.params
        []
        >>> i.output
        x
        """
        self.input = input
        self.output = input
        self.output_shape = input_shape
        self.params = []
