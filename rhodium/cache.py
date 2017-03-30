# Copyright 2015-2016 David Hadka
#
# This file is part of Rhodium, a Python module for robust decision making and
# exploratory modeling.
#
# Rhodium is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Rhodium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Rhodium.  If not, see <http://www.gnu.org/licenses/>.
import os
import sys
import shelve
import atexit
import pickle
import inspect
import warnings
import functools
import collections

_CACHE_FILE = None
_CACHE_CLEAR = False
_CACHE_WARNINGS = True
_CACHE = None

def setup_cache(file=None, clear=False, warn=True):
    global _CACHE_FILE
    global _CACHE_CLEAR
    global _CACHE_WARNINGS
    global _CACHE
    
    if _CACHE:
        if warn:
            warnings.warn("cache already opened", UserWarning)
        return
    
    _CACHE_FILE = file
    _CACHE_CLEAR = clear
    _CACHE_WARNINGS = warn
  
def _do_open():
    global _CACHE_FILE
    global _CACHE_CLEAR
    global _CACHE
    
    if _CACHE is not None:
        return
    
    if not _CACHE_FILE:
        _CACHE_FILE = "rhodium.cache"

    _CACHE = shelve.open(_CACHE_FILE, flag='n' if _CACHE_CLEAR else 'c')
    
    atexit.register(lambda: _do_close())
    
def _do_close():
    if _CACHE is not None:
        _CACHE.close()
        
def cached(func):
    
    @functools.wraps(func)
    def inner(*args, **kwargs):
        try:
            arguments = (args, kwargs)
            
            if sys.version_info[0] == 2:
                arguments_str = pickle.dumps(arguments)
            else:
                arguments_str = pickle.dumps(arguments, protocol=0).decode('ascii')
            
            return cache(func.__name__ + arguments_str, lambda: func(*args, **kwargs))
        except pickle.PickleError:
            if _CACHE_WARNINGS:
                warnings.warn("arguments not pickleable, unable to cache value", UserWarning)
            return func(*args, **kwargs)
    
    return inner

def clear_cache(*args):
    _do_open()
        
    for arg in args:
        if arg in _CACHE:
            del _CACHE[arg]
  
def cache(key, value):
    """Simple file cache using shelve.
    
    Aims to be a very simple wrapper for shelve.  Suppose we are performing
    an expensive function evaluation we want to cache:
    
        data = compute(args)
        
    we can cache the result and load it if available:
    
        data = cache("data", lambda: compute(args))
        
    Note the use of lambda to enable lazy evaluation of the function.
    """
    _do_open()

    if not inspect.isfunction(value):
        if _CACHE_WARNINGS:
            warnings.warn("cache value should be a function to enable lazy evaluation", UserWarning)
        
    if key not in _CACHE:
        if inspect.isfunction(value):
            _CACHE[key] = value()
        else:
            _CACHE[key] = value
        
    return _CACHE[key]
