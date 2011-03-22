from functools import update_wrapper

__all__ = ['add_pre', 'add_post', 'remove_pre', 'remove_post', 'dehook']

def add_pre(env, name, f):
    r"""
    Add a function to be called before env.name.

    The function will recieve the exact arguments that the function
    will have, including defaults.
    """
    hookfn = _hook(env, name)
    getattr(hookfn, '__hook.pre').append(f)

def add_post(env, name, f):
    r"""
    Add a function to be called after env.name.
    
    `f` will get as first argument a tuple with (result, exception)
    from the function.  One of the elements will be None depending on
    whether the function raised or not.  The subsequent arguments will
    be exactly the same that the function got, including defaults.
    """
    hookfn = _hook(env, name)
    getattr(hookfn, '__hook.post').append(f)

def remove_pre(env, name, f):
    r"""
    Remove `f` as a pre hook from env.name, if present.
    """
    fn = getattr(env, name)
    if hasattr(fn, '__hook.pre'):
        try:
            getattr(fn, '__hook.pre').remove(f)
        except ValueError:
            pass

def remove_post(env, name, f):
    r"""
    Remove `f` as a post hook from env.name, if present.
    """
    fn = getattr(env, name)
    if hasattr(fn, '__hook.post'):
        try:
            getattr(fn, '__hook.post').remove(f)
        except ValueError:
            pass

def dehook(env, name):
    r""" 
    Clears all hooks on env.name and removes the hook machinery, if
    present.
    """
    _dehook(env, name)

hooked_template = r"""
def %(name)s(%(argsdecl)s):
    for pre in getattr(%(name)s, '__hook.pre'):
        pre(%(argnames)s)
    try:
        res = getattr(%(name)s, '__hook.orig')(%(argnames)s)
    except e:
        for post in getattr(%(name)s, '__hook.post'):
            post((None, e), %(argnames)s)
        raise
    else:
        for post in getattr(%(name)s, '__hook.post'):
            post((res, None), %(argnames)s)
    return res
"""

def _classr(object):
    def __init__(self, repr):
        self.repr = repr
    def __repr__(self):
        return self.repr

def _hook(env, name):
    fn = env.__dict__[name]
    if hasattr(fn, '__hook.orig'):
        return fn

    globs = fn.func_globals.copy()

    argspec = inspect.getargspec(fn)
    defaults = argspec[3]
    if defaults:
        newdef = []
        for i, v in enumerate(defaults):
            name = '__default_arg_value_%d__'%(i,)
            globs[name] = v
            newdef.append(_classr(name))
        argspec = argspec[:3] + (tuple(newdefs),)

    nodefs = argspec[:3]+(None,)
    code = hooked_template % dict(name=fn.func_name,
                                  argnames=inspect.formatargspec(*nodefs),
                                  argsdecl=inspect.formatargspec(*argspec))

    locs = dict()

    eval(code, globs, locs)
    wrapfn = locs[fn.func_name]
    setattr(wrapfn, '__hook.orig', fn)
    setattr(wrapfn, '__hook.pre', [])
    setattr(wrapfn, '__hook.post', [])
    
    functools.update_wrapper(wrapfn, fn)
    
    env.__dict__[name] = wrapfn
    return wrapfn

def _dehook(env, name):
    fn = env.__dict__[name]
    if hasattr(fn, '__hook.orig'):
        origfn = getattr(fn, '__hook.orig')
        env.__dict__[name] = origfn
