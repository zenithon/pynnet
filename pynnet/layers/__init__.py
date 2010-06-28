r"""
Module grouping the different layer implementations.

All layers should be objects that have the following attributes:

`input` -- the input expression used for the layer
`output` -- the output of the layer
`output_shape` -- the shape of the output (may be None)
`params` -- parameters that are to be adjusted with the gradient

defined after the build() method is called.

The build() method has this signature:

def build(self, input, input_shape=None):
   # do stuff

The `input` attribute should be a copy of the input parameter passed
to build().

The `input_shape` parameter must be a complete shape (no None
elements) or None.  If it is provided, the `output_shape` must be
defined to a complete shape.  Otherwise it may be None.

Any additional parameters for the layer should be collected at object
construction time.

Most layers define additional attributes that influence their output.
Look at the class documentation for a list of the relevant attributes.
Note that you will have to rebuild the network in order to use the new
parameters you may have set and that any already built theano function
will not be updated.
"""
from simple import *
from conv import *
from autoencoder import *
from composite import *
from recurrent import *
from lstm import *
