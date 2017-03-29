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
from __future__ import division, print_function, absolute_import

import six
import random
import inspect
import numbers
import functools
from rhodium import *

MINIMIZE = Response.MINIMIZE
MAXIMIZE = Response.MAXIMIZE
INFO = Response.INFO
IGNORE = Response.IGNORE

class UnnamedObject(object):
    
    def __init__(self, constructor, *args, **kwargs):
        super(UnnamedObject, self).__init__()
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs
        
    def convert(self, name):
        return self.constructor(name, *self.args, **self.kwargs)

class Real(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Real, self).__init__(RealLever, *args, **kwargs)
        
class Integer(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Integer, self).__init__(IntegerLever, *args, **kwargs)
        
class Categorical(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Categorical, self).__init__(CategoricalLever, *args, **kwargs)
        
class Permutation(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Permutation, self).__init__(PermutationLever, *args, **kwargs)
        
class Subset(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Subset, self).__init__(SubsetLever, *args, **kwargs)

class Uniform(float, UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Uniform, self).__init__(UniformUncertainty, *args, **kwargs)
        
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))
        
class Normal(float, UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Normal, self).__init__(NormalUncertainty, *args, **kwargs)
        
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))
        
class LogNormal(float, UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(LogNormal, self).__init__(LogNormalUncertainty, *args, **kwargs)
        
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))
        
        
class Minimize(Response):
    
    def __init__(self, name, **kwargs):
        super(Minimize, self).__init__(name, type=Response.MINIMIZE, **kwargs)
        
class Maximize(Response):
    
    def __init__(self, name, **kwargs):
        super(Maximize, self).__init__(name, type=Response.MAXIMIZE, **kwargs)
     
class Info(Response):
    
    def __init__(self, name, **kwargs):
        super(Info, self).__init__(name, type=Response.INFO, **kwargs)
        
class Ignore(Response):
    
    def __init__(self, **kwargs):
        super(Ignore, self).__init__("Ignored" + str(random.randint()), type=Response.IGNORE, **kwargs)

class CallableModel(Model):
    
    def __init__(self, function):
        super(CallableModel, self).__init__(function)
    
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class RhodiumModel(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):        
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
        argspec = inspect.getargspec(f)
        nargs = len(argspec.args)
        ndefs = len(argspec.defaults)
            
        for i, k in enumerate(argspec.args):
            if i >= nargs-ndefs:
                # has default value, determine if special rhodium annotation
                def_val = argspec.defaults[i-(nargs-ndefs)]
                
                if isinstance(def_val, UnnamedObject):
                    named_obj = def_val.convert(k)
                    
                    if isinstance(named_obj, Lever):
                        f.rhodium_model.levers[k] = named_obj
                    elif isinstance(named_obj, Uncertainty):
                        f.rhodium_model.uncertainties[k] = named_obj
            
            if k not in f.rhodium_model.parameters:
                f.rhodium_model.parameters[k] = Parameter(k)
            
        return f.rhodium_model
    
class Parameters(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
        
        for v in self.args:
            if isinstance(v, six.string_types):
                f.rhodium_model.parameters[v] = Parameter(v)
            elif isinstance(v, Parameter):
                f.rhodium_model.parameters[v.name] = v
            else:
                raise ValueError("arg must be a string or Parameter")
        
        for k, v in six.iteritems(self.kwargs):
            if isinstance(v, dict):
                f.rhodium_model.parameters[k] = Parameter(k, **v)
            elif isinstance(v, (list, tuple)):
                f.rhodium_model.parameters[k] = Parameter(k, *v)
            else:
                f.rhodium_model.parameters[k] = Parameter(k, v)
        
        return f
    
class Responses(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
#         if len(self.args) == 2 and isinstance(self.args[0], (list, tuple)) and isinstance(self.args[1], (list, tuple)):
#             for i, v in enumerate(self.args[0]):
#                 f.rhodium_model.responses[v] = Response(v, self.args[1][i if i < len(self.args[1]) else -1])
#         elif len(self.args) == 1 and isinstance(self.args[0], (list, tuple)):
#             for v in self.args[0]:
#                 f.rhodiuim_model.responses[v] = Response(v, Response.INFO)
#         else:
#             for v in self.args:
#                 if isinstance(v, six.string_types):
#                     f.rhodium_model.responses[v] = Response(v, Response.INFO)
#                 elif isinstance(v, Response):
#                     f.rhodium_model.responses[v.name] = v
#                 else:
#                     raise ValueError("arg must be a string or Response")

        for v in self.args:
            if isinstance(v, six.string_types):
                f.rhodium_model.responses[v] = Response(v, Response.INFO)
            elif isinstance(v, Response):
                f.rhodium_model.responses[v.name] = v
            else:
                raise ValueError("arg must be a string or Response")
        
        for k, v in six.iteritems(self.kwargs):
            if isinstance(v, UnnamedObject):
                f.rhodium_model.responses[k] = v.convert(k)
            elif isinstance(v, dict):
                f.rhodium_model.responses[k] = Response(k, **v)
            elif isinstance(v, (list, tuple)):
                f.rhodium_model.responses[k] = Response(k, *v)
            else:
                f.rhodium_model.responses[k] = Response(k, v)
        
        return f
    
class Constraints(object):
    
    def __init__(self, *args):
        self.args = args
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
        for v in self.args:
            if isinstance(v, Constraint):
                f.rhodium_model.constraints.append(v)
            elif isinstance(v, six.string_types) or callable(v):
                f.rhodium_model.constraints.append(Constraint(v))
            else:
                raise ValueError("arg must be a Constraint, string, or callable")
                
        return f
    
class Levers(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
        for v in self.args:
            if isinstance(v, Lever):
                f.rhodium_model.levers[v.name] = v
            else:
                raise ValueError("arg must be a Lever")
            
        for k, v in six.iteritems(self.kwargs):
            if isinstance(v, UnnamedObject):
                f.rhodium_model.levers[k] = v.convert(k)
            else:
                raise ValueError("kwarg must be a Lever")
                
        return f
    
class Uncertainties(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
        for v in self.args:
            if isinstance(v, Uncertainty):
                f.rhodium_model.uncertainties[v.name] = v
            else:
                raise ValueError("arg must be an Uncertainty")
            
        for k, v in six.iteritems(self.kwargs):
            if isinstance(v, UnnamedObject):
                f.rhodium_model.uncertainties[k] = v.convert(k)
            else:
                raise ValueError("kwarg must be an Uncertainty")
                
        return f