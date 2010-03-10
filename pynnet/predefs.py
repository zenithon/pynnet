from pynnet import *

def LogisticRegression(n_in, n_out):
    return NNet([Layer(n_in, n_out, activation=nlins.softmax)],
                error=errors.nll)
