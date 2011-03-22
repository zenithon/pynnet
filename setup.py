#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pynnet',
      version='0.3.0dev',
      description='python neural network library',
      author='Arnaud Bergeron',
      author_email='abergeron@gmail.com',
      url='http://code.google.com/p/pynnet',
      packages=find_packages(),
      requires=['numpy', 'theano'],
      license='MIT'
      classifiers = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License"
        ]
      )
