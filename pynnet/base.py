from __future__ import with_statement

__all__ = ['BaseObject', 'theano', 'T', 'numpy', 'load', 'loadf',
           'test_saveload']

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

from contextlib import closing
import tempfile, os, zipfile

import numpy
import theano
import theano.tensor as T

def zipadd(fn, zf, name):
    r"""
    Calls `fn` with a file pointer and adds the content of the file to
    `zf` (which is a zip archive) under name `name`

    :notests:
    """
    try:
        fid, fname = tempfile.mkstemp()
        fp = os.fdopen(fid, 'wb')
        fn(fp)
        fp.close()
        zf.write(fname, arcname=name)
    finally:
        if fp:
            fp.close()
            os.remove(fname)

class PersSave(object):
    def __init__(self, zf):
        self.zf = zf
        self.count = 0
    def __call__(self, obj):
        if isinstance(obj, numpy.ndarray):
            name = 'array-'+str(self.count)
            self.count += 1
            def fn(fp):
                numpy.lib.format.write_array(fp, obj)
            zipadd(fn, self.zf, name)
            return name
        else:
            return None

class PersLoad(object):
    def __init__(self, zf):
        self.zf = zf
    def __call__(self,id):
        return numpy.lib.format.read_array(self.zf.open(id))

def zsave(obj, file):
    r"""
    Saves `obj` to `file` (which is a file-like object).

    :notests:
    """
    with closing(zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)) as zf:
        def fn(fp):
            p = pickle.Pickler(fp, 2)
            p.persistent_id = PersSave(zf)
            p.dump(obj)
        zipadd(fn, zf, 'pkl')

def zload(file):
    r"""
    Loads a save created with `zsave`.

    :notests:
    """
    with closing(zipfile.ZipFile(file, 'r')) as zf:
        p = pickle.Unpickler(StringIO.StringIO(zf.open('pkl').read()))
        p.persistent_load = PersLoad(zf)
        return p.load()

def test_saveload(obj):
    r"""
    Saves and loads `obj` and returns the loaded copy.
    
    :notests:
    """
    import StringIO
    f = StringIO.StringIO()
    obj.savef(f)
    f2 = StringIO.StringIO(f.getvalue())
    f.close()
    obj2 = loadf(f2)
    f2.close()
    return obj2

class BaseObject(object):
    def save(self, fname):
        r"""
        Save the object to disk.

        The named file will be created if not present and overwritten
        if present.

        Do NOT override this method, implement __getstate__ and
        __setstate__ as per pickle rules.
        """
        with open(fname, 'wb') as f:
            self.savef(f)

    def savef(self, f):
        r"""
        Save the object to the file-like object `f`.

        Do NOT override this method, implement __getstate__ and
        __setstate__ as per pickle rules.
        """
        zsave(self, f)
    
def load(fname):
    r"""
    Load an object from a save file.
    """
    with open(fname, 'rb') as f:
        return loadf(f)

def loadf(f):
    r"""
    Loads an object from the file-like object `f`.    
    """
    return zload(f)
