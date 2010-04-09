import doctest, sys, pkgutil, types, pynnet

def runTests(mod):
    for (_, name, ispkg) in pkgutil.walk_packages(mod.__path__, mod.__name__+'.'):
        if not ispkg:
            test(name)
            cover(name)

def test(name):
    predefs = dict(pynnet.__dict__)
    options = doctest.ELLIPSIS or doctest.DONT_ACCEPT_TRUE_FOR_1
    print "Testing:", name
    __import__(name)
    doctest.testmod(sys.modules[name], extraglobs=predefs, optionflags=options)

def cover(name):
    __import__(name)
    for meth in methods_of(sys.modules[name]):
        if meth.__name__ in ['_save_', '_load_', '__str__']:
            continue
        if meth.__doc__ is None:
            print "*** No doc for:", meth.__name__
            
def methods_of(obj, mod=None):
    if isinstance(obj, types.ModuleType):
        mod = obj
    if mod is None:
        raise ValueError('must be called with a module')
    for aname in dir(obj):
        attr = getattr(obj, aname)
        if not hasattr(attr, '__module__') or attr.__module__ != mod.__name__:
            continue
        if isinstance(attr, type):
            for meth in methods_of(attr, mod):
                yield meth
        if isinstance(attr, (types.FunctionType, types.UnboundMethodType)):
            yield attr

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
                cover(mod)
    else:
        runTests(pynnet)
