r"""
Module grouping the different node implementations.

All nodes must have the following attributes:

`inputs` -- the list of input nodes
`output` -- the output expression of the node
`params` -- parameters that are to be adjusted with the gradient

To help this you can inherit from the BaseNode class which imposes the
following requirements:

- You must have a `transform` method that takes the inputs for the
  node and returns the output expression

def transform(self, *inputs):
    # do stuff
    ...
    return output_expr

- You may have a `local_params` attribute that lists the gradient
  adjustable parameters for this node.  You must define this after
  calling the superclass constructor.

In return it provides `output` and `params` attributes which are
automatically computed when needed.
"""
from simple import *
from conv import *
from autoencoder import *
from recurrent import *
from lstm import *
