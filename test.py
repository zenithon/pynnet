import doctest, sys, pkgutil, pynnet

def runTests(mod):
    for (_, name, ispkg) in pkgutil.walk_packages(mod.__path__, mod.__name__+'.'):
        if not ispkg:
            test(name)

def test(name):
    predefs = dict(pynnet.__dict__)
    options = doctest.ELLIPSIS or doctest.DONT_ACCEPT_TRUE_FOR_1
    print "Testing:", name
    __import__(name)
    doctest.testmod(sys.modules[name], extraglobs=predefs, optionflags=options)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for mod in sys.argv[1:]:
            if mod.endswith('.py'):
                mod = mod[:-3]
            if mod.endswith('/') or mod.endswith('\\'):
                mod = mod[:-1]
            mod = mod.replace('/', '.')
            mod = mod.replace('\\', '.')
            __import__(mod)
            mm = sys.modules[mod]
            if hasattr(mm, '__path__'):
                runTests(mm)
            else:
                test(mod)
    else:
        runTests(pynnet)
