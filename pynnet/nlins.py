import theano.tensor as T

__all__ = ['tanh', 'sigmoid', 'softmax', 'none']

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def softmax(x):
    return T.nnet.softmax(x)

def none(x):
    return x
