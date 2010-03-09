r"""
Module grouping the different layer implementations.

All layers should be objects that have the following attributes:

`input` -- the input expression used for the layer
`output` -- the output of the layer
`params` -- parameters that are to be adjusted with the gradient

defined after the build() method is called.

The build() method has this signature:

def build(self, input):
   # do stuff

The `input` attribute should be a copy of the input parameter passed
to build().

Any additional parameters for the layer should be collected at object
construction time.

Some specialized layers may define additional attributes.
"""
from hidden import *
from conv import *
