#!/usr/bin/env python

from distutils.core import setup

setup(name='pynnet',
      version='0.2.0dev',
      description='python neural network library',
      author='Arnaud Bergeron',
      author_email='abergeron@gmail.com',
      url='http://code.google.com/p/pynnet',
      packages=['pynnet', 'pynnet.layers'],
      requires=['numpy', 'theano'],
      license='MIT'
      )
