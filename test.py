import doctest, sys, os, pkgutil, types, pynnet

nullfile = file(os.devnull, "wb")

def runTests(mod):
    for (_, name, ispkg) in pkgutil.walk_packages(mod.__path__, mod.__name__+'.'):
        if not ispkg:
            test(name)
            cover(name)

def test(name):
    predefs = dict(pynnet.__dict__)
    options = doctest.ELLIPSIS or doctest.DONT_ACCEPT_TRUE_FOR_1
    print >>sys.stderr, "Testing:", name
    __import__(name)
    doctest.testmod(sys.modules[name], extraglobs=predefs, optionflags=options)

def cover(name):
    __import__(name)
    for meth in methods_of(sys.modules[name]):
        if meth.__name__ in ['__str__']:
            continue
        if meth.__doc__ is None:
            print "*** No doc for:", meth.__name__
            
def methods_of(obj, mod=None):
    if isinstance(obj, types.ModuleType):
        mod = obj
    if mod is None:
        raise ValueError('must be called with a module')
    for aname in dir(obj):
        if aname in ('__abstractmethods__',):
            continue
        attr = getattr(obj, aname)
        if not hasattr(attr, '__module__') or attr.__module__ != mod.__name__:
            continue
        if isinstance(attr, type):
            for meth in methods_of(attr, mod):
                yield meth
        if isinstance(attr, (types.FunctionType, types.UnboundMethodType)):
            yield attr

def test_example(file):
    fail = False
    env = dict()
    print "Trying:", file,
    sys.stdout.flush()
    stdout = sys.stdout
    sys.stdout = nullfile
    try:
        execfile(file, env)
    except Exception, e:
        fail = e
    finally:
        sys.stdout = stdout

    if fail:
        print "FAILED"
        print type(fail), fail
    else:
        print "OK"

def try_examples(dir):
    for file in os.listdir(dir):
        if file[-3:] == '.py':
            test_example(os.path.join(dir, file))

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
        try_examples('examples')
