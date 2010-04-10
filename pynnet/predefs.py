from pynnet import *

def LogisticRegression(n_in, n_out):
    r"""
    :nodoc:
    """
    return NNet([Layer(n_in, n_out, activation=nlins.softmax)],
                error=errors.nll)
