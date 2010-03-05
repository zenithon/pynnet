from __future__ import with_statement

__all__ = ['BaseObject', 'theano', 'numpy', 'pickle', 'layer', 'propres']

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy
import theano

try:
    from collections import namedtuple
except ImportError:
    # The 'namedtuple' code was stolen from the python 2.6.1 library so:
    #
    # Copyright (c) 2001-2009 Python Software Foundation; All Rights Reserved
    #
    # PSF LICENSE AGREEMENT FOR PYTHON 2.6.1
    #
    # 1. This LICENSE AGREEMENT is between the Python Software Foundation
    #    ("PSF"), and the Individual or Organization ("Licensee") accessing 
    #    and otherwise using Python 2.6.1 software in source or binary form
    #    and its associated documentation.
    #
    # 2. Subject to the terms and conditions of this License Agreement, PSF
    #    hereby grants Licensee a nonexclusive, royalty-free, world-wide 
    #    license to reproduce, analyze, test, perform and/or display publicly,
    #    prepare derivative works, distribute, and otherwise use Python 2.6.1
    #    alone or in any derivative version, provided, however, that PSF's 
    #    License Agreement and PSF's notice of copyright, i.e., 
    # "Copyright (c) 2001-2009 Python Software Foundation; All Rights Reserved"
    #    are retained in Python 2.6.1 alone or in any derivative version 
    #    prepared by Licensee.
    #
    # 3. In the event Licensee prepares a derivative work that is based on or 
    #    incorporates Python 2.6.1 or any part thereof, and wants to make the 
    #    derivative work available to others as provided herein, then Licensee 
    #    hereby agrees to include in any such work a brief summary of the 
    #    changes made to Python 2.6.1.
    #
    # 4. PSF is making Python 2.6.1 available to Licensee on an "AS IS" basis.
    #    PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY
    #    WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY 
    #    REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY 
    #    PARTICULAR PURPOSE OR THAT THE USE OF PYTHON 2.6.1 WILL NOT INFRINGE 
    #    ANY THIRD PARTY RIGHTS.
    #
    # 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 
    #    2.6.1 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS 
    #    AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 
    #    2.6.1, OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY
    #    THEREOF.
    #
    # 6. This License Agreement will automatically terminate upon a material
    #    breach of its terms and conditions.
    #
    # 7. Nothing in this License Agreement shall be deemed to create any
    #    relationship of agency, partnership, or joint venture between PSF and
    #    Licensee. This License Agreement does not grant permission to use PSF
    #    trademarks or trade name in a trademark sense to endorse or promote
    #    products or services of Licensee, or any third party.
    #
    # 8. By copying, installing or otherwise using Python 2.6.1, Licensee
    #    agrees to be bound by the terms and conditions of this License 
    #    Agreement.

    from operator import itemgetter as _itemgetter
    from keyword import iskeyword as _iskeyword
    import sys as _sys

    def namedtuple(typename, field_names, verbose=False):
        """Returns a new subclass of tuple with named fields.
        
        >>> Point = namedtuple('Point', 'x y')
        >>> Point.__doc__                   # docstring for the new class
        'Point(x, y)'
        >>> p = Point(11, y=22)             # instantiate with positional args or keywords
        >>> p[0] + p[1]                     # indexable like a plain tuple
        33
        >>> x, y = p                        # unpack like a regular tuple
        >>> x, y
        (11, 22)
        >>> p.x + p.y                       # fields also accessable by name
        33
        >>> d = p._asdict()                 # convert to a dictionary
        >>> d['x']
        11
        >>> Point(**d)                      # convert from a dictionary
        Point(x=11, y=22)
        >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
        Point(x=100, y=22)
        """
        # Parse and validate the field names.  Validation serves two purposes,
        # generating informative error messages and preventing template injection attacks.
        if isinstance(field_names, basestring):
            field_names = field_names.replace(',', ' ').split() # names separated by whitespace and/or commas
            field_names = tuple(map(str, field_names))
        for name in (typename,) + field_names:
            if not all(c.isalnum() or c=='_' for c in name):
                raise ValueError('Type names and field names can only contain alphanumeric characters and underscores: %r' % name)
            if _iskeyword(name):
                raise ValueError('Type names and field names cannot be a keyword: %r' % name)
            if name[0].isdigit():
                raise ValueError('Type names and field names cannot start with a number: %r' % name)
        seen_names = set()
        for name in field_names:
            if name.startswith('_'):
                raise ValueError('Field names cannot start with an underscore: %r' % name)
            if name in seen_names:
                raise ValueError('Encountered duplicate field name: %r' % name)
            seen_names.add(name)

        # Create and fill-in the class template
        numfields = len(field_names)
        argtxt = repr(field_names).replace("'", "")[1:-1]   # tuple repr without parens or quotes
        reprtxt = ', '.join('%s=%%r' % name for name in field_names)
        dicttxt = ', '.join('%r: t[%d]' % (name, pos) for pos, name in enumerate(field_names))
        template = '''class %(typename)s(tuple):
        '%(typename)s(%(argtxt)s)' \n
        __slots__ = () \n
        _fields = %(field_names)r \n
        def __new__(cls, %(argtxt)s):
            return tuple.__new__(cls, (%(argtxt)s)) \n
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new %(typename)s object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != %(numfields)d:
                raise TypeError('Expected %(numfields)d arguments, got %%d' %% len(result))
            return result \n
        def __repr__(self):
            return '%(typename)s(%(reprtxt)s)' %% self \n
        def _asdict(t):
            'Return a new dict which maps field names to their values'
            return {%(dicttxt)s} \n
        def _replace(self, **kwds):
            'Return a new %(typename)s object replacing specified fields with new values'
            result = self._make(map(kwds.pop, %(field_names)r, self))
            if kwds:
                raise ValueError('Got unexpected field names: %%r' %% kwds.keys())
            return result \n
        def __getnewargs__(self):
            return tuple(self) \n\n''' % locals()
        for i, name in enumerate(field_names):
            template += '        %s = property(itemgetter(%d))\n' % (name, i)
        if verbose:
            print template

        # Execute the template string in a temporary namespace and
        # support tracing utilities by setting a value for frame.f_globals['__name__']
        namespace = dict(itemgetter=_itemgetter, __name__='namedtuple_%s' % typename)
        try:
            exec template in namespace
        except SyntaxError, e:
            raise SyntaxError(e.message + ':\n' + template)
        result = namespace[typename]

        # For pickling to work, the __module__ variable needs to be set to the frame
        # where the named tuple is created.  Bypass this step in enviroments where
        # sys._getframe is not defined (Jython for example).
        if hasattr(_sys, '_getframe'):
            result.__module__ = _sys._getframe(1).f_globals['__name__']

        return result

layer = namedtuple('layer', 'W b')
propres = namedtuple('propres', 'acts outs')

class BaseObject(object):
    def _save_(self, file):
        r"""
        Save the state to a file.

        This receives a file object argument and should dump the
        variables and maybe a tag to the file stream.

        You can use the `pickle.dump()` method to save objects, but
        please use `numpy.save()`(and not `numpy.savez()`) to save
        numpy arrays if you must.  `numpy` and `pickle` do not mix
        well.  Do not write to the file before the position of the
        file pointer when you reveived the file objet.  Also leave the
        file pointer at the end of the written data when you are
        finished.  The `numpy` and `pickle` methods do this
        automatically.

        It is also generally a good idea to write some kind of tag to
        uniquely identify your class and prevent the loading of bad
        data.  This tag can also be used to identify the format
        version in case you ever decide to change it.

        You only need to care about the variables you define yourself.
        In particular do not call the `_save_()` method of your
        parent(s) class(es).  
        """
        file.write("SOSV1")

    def save(self, fname):
        r"""
        Save the object to disk.

        The named file will be created if not present and overwritten
        if present.

        Do NOT override this method, implement a `_save_()`
        method for your classes.
        """
        if hasattr(self, '_vitual'):
            raise ValueError('Cannot save a virtual object.  Save the parent instead.')
        
        if type(fname) is file:
            self.__save(fname)
        else:
            with open(fname, 'wb') as f:
                self.__save(f)

    def __save(self, f):
        for C in reversed(type(self).__mro__):
            if hasattr(C, '_save_'):
                C._save_(self, f)
    
    @classmethod
    def load(cls, fname):
        r"""
        Load an object from a save file.

        The resulting object will have the same class as the calling
        class of this function.  If the saved object in the file is
        not of the appropriate class exceptions may be raised.

        Do NOT override this method, implement a `_load_()`
        method for your classes.

        Do NOT rely on being able to load an objet as a different
        class than the one it was before save() since that possibility
        may go away in the future.
        """
        obj = object.__new__(cls)
        if type(fname) is file:
            cls.__load(fname, obj)
        else:
            with open(fname, 'rb') as f:
                cls.__load(f, obj)
        return obj

    @classmethod
    def __load(cls, f, obj):
        for C in reversed(type(obj).__mro__):
            if hasattr(C, '_load_'):
                C._load_(obj, f)
    
    def _load_(self, file):
        r"""
        Load the state from a file.

        You should load what you saved in the `_save_()` method.  Be
        careful to leave the file pointer at the end of your saved
        data.  The `numpy` and `pickle` methods do this automatically.
        """
        str = file.read(5)
        if str != "SOSV1":
            raise ValueError('Not a save file of file is corrupted')
