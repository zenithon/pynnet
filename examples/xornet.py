# This is a test file for simplenet doing a xor regression test
# It is an experiment in which we try to make a network learn the
# patterns of input and output for the xor function.

import theano
from pynnet import *

# We initialize an MLP with one hidden layer of two units.
xornet = NNet([SimpleLayer(2,2), SimpleLayer(2,1)], error=errors.mse)

sx = theano.tensor.matrix('x')
sy = theano.tensor.matrix('y')

# This call actually assembles the expressions for .output, .cost and .params
# from the layers we have defined in the network.
xornet.build(sx, sy)

# We can then build functions from these expressions to use our network
print "Compiling theano functions, this make take some time."
eval = theano.function([sx], xornet.output)
test = theano.function([sx, sy], xornet.cost)
train = theano.function([sx, sy], xornet.cost, 
                        updates=trainers.get_updates(xornet.params, xornet.cost, alpha=0.01))


x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# At the start the error is terrible
print "Error at start:", test(x, y)

# So we train a little
for i in range(10):
    train(x, y)

# Now the error should be a bit better
print "Error after 10:", test(x, y)

# And we train more
for i in range(10000):
    train(x, y)

# Now the error should be really low
print "Error after 10000:", test(x, y)

# We can look at the actual output of the network
print "Output for [0, 1]:", eval([[0, 1]])
