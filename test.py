import doctest

def runTests(options = doctest.ELLIPSIS or doctest.DONT_ACCEPT_TRUE_FOR_1):
    import pynnet
    
    predefs = pynnet.__dict__
    
    for mod in [pynnet.autoencoder, pynnet.base, pynnet.errors, pynnet.net, pynnet.nlins, pynnet.simplenet, pynnet.trainers]:
        doctest.testmod(mod, extraglobs=predefs, optionflags=options)

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    runTests()
