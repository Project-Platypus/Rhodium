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

import ast
import six
import math
import inspect
import random
import operator
import scipy.stats as stats
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from platypus import Real

class RhodiumError(Exception):
    pass

class NamedObject(object):
    """Object with a name."""
    
    def __init__(self, name):
        super(NamedObject, self).__init__()
        self.name = name

class Parameter(NamedObject):
    """Defines a model parameter (i.e., input).
    
    Defines a model input (i.e., input) and an optional default value.  The
    name must be identical to the keyword argument to the method defining the
    model.
    """ 
    
    def __init__(self, name, default_value = None):
        super(Parameter, self).__init__(name)
        self.default_value = default_value
        
class Response(NamedObject):
    """Defines a model response (i.e., output).
    
    Defines a model response (i.e., output) and its type.  The type can be
    MINIMIZE, MAXIMIZE, or INFO.  If MINIMIZE or MAXIMIZE, then the response
    may be used during optimization.  If INFO, the default, the response is
    purely for informative purposes (e.g., for generating plots) but does not
    participate in optimization.
    """
    
    MINIMIZE = -1
    MAXIMIZE = 1
    INFO = 0
    
    def __init__(self, name, type = INFO):
        super(Response, self).__init__(name)
        self.type = type
        
_eval_env = {}
module = __import__("math", fromlist=[''])

for name in dir(module):
    if not name.startswith("_"):
        _eval_env[name] = getattr(module, name)
        
class Constraint(object):
    """Defines model constraints.
    
    Defines constraints that must be satisfied in order for a policy to be
    considered feasible.  This is often called a "hard constraint."
    Constraints can either be defined using a valid Python expression that
    references any parameters or responses (passed as a string) or a function
    given a dict of the parameters and responses.
    """
    
    def __init__(self, expr):
        super(Constraint, self).__init__()
        self.expr = expr
        
        if isinstance(expr, str):
            self._convert()
        
    def _convert(self):
        """Attempts to convert expression to distance function.
        
        Constraints are often expressed as inequalities, such as x < 5, meaning
        that a policy is feasible if the value of x is less than 5.  It is
        sometimes useful to know how far a policy is from a feasibility
        threshold.  For example, x = 7 is closer to the feasibility threshold
        than x = 15.
        
        This method attempts to convert a comparison expression to a distance
        expression by manipulating the AST.  If successful, this method creates
        the _distance attribute.  Even if this method is successful, the
        generated expression may not be valid.
        """
        root = ast.parse(self.expr, mode="eval")
        
        if isinstance(root.body, ast.Compare) and len(root.body.ops) == 1:
            left_expr = root.body.left
            right_expr = root.body.comparators[0]
            
            distance_expr = ast.Expression(ast.BinOp(left_expr,
                                                     ast.Sub(),
                                                     right_expr))
            
            ast.fix_missing_locations(distance_expr)
            self._distance = compile(distance_expr, "<AST>", "eval")

    def is_feasible(self, env):
        tmp_env = {}
        tmp_env.update(_eval_env)
        tmp_env.update(env)
        
        if isinstance(self.expr, str):
            return eval(self.expr, {}, tmp_env)
        else:
            return self.expr(tmp_env)
    
    def distance(self, env):
        """Returns the distance to the feasibility threshold."""
        if self.is_feasible(env):
            return 0.0
        elif hasattr(self, "_distance"):
            try:
                tmp_env = {}
                tmp_env.update(_eval_env)
                tmp_env.update(env)
                return abs(eval(self._distance, {}, tmp_env)) + 0.001
            except:
                return 1.0
        else:
            return 1.0
        
class Lever(NamedObject):
    """Defines an adjustable lever that controls a model parameter.
    
    Model parameters can either be constant, controlled by a lever, or
    subject to uncertainty.  The lever defines the available options for
    a given design factor.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, name):
        super(Lever, self).__init__(name)
        
    @abstractmethod
    def to_variables(self):
        raise NotImplementedError("method not implemented")
    
class RealLever(Lever):
    
    def __init__(self, name, min_value, max_value, length = 1):
        super(RealLever, self).__init__(name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.length = length
        
    def to_variables(self):
        return [Real(self.min_value, self.max_value) for _ in range(self.length)]
        
class Uncertainty(NamedObject):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, name):
        super(Uncertainty, self).__init__(name)
        
    @abstractmethod    
    def levels(self, nlevels):
        raise NotImplementedError("method not implemented")
    
    @abstractmethod    
    def ppf(self, x):
        raise NotImplementedError("method not implemented")
        
class UniformUncertainty(Uncertainty):
    
    def __init__(self, name, min_value, max_value):
        super(UniformUncertainty, self).__init__(name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def levels(self, nlevels):
        d = (self.max_value - self.min_value) / nlevels
        result = []
        
        for i in range(nlevels):
            result.append(self.min_value + random.uniform(i*d, (i+1)*d))
        
        return result
    
    def ppf(self, x):
        self.min_value + x*(self.max_value - self.min_value)
    
class NormalUncertainty(Uncertainty):
    
    def __init__(self, name, mean, stdev):
        super(NormalUncertainty, self).__init__(name)
        self.mean = float(mean)
        self.stdev = float(stdev)
        
    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, 0.0, 1.0).levels(nlevels)
        return stats.norm.ppf(ulevels, self.mean, self.stdev)
    
    def ppf(self, x):
        return stats.norm.ppf(x, self.mean, self.stdev)
    
class LogNormalUncertainty(Uncertainty):
    
    def __init__(self, name, mu, sigma):
        super(LogNormalUncertainty, self).__init__(name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        
    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, 0.0, 1.0).levels(nlevels)
        return self.mu*stats.lognorm.ppf(ulevels, self.sigma)
    
    def ppf(self, x):
        return self.mu*stats.lognorm.ppf(x, self.sigma)

class IntegerUncertainty(Uncertainty):
    
    def __init__(self, name, min_value, max_value):
        super(IntegerUncertainty, self).__init__(name)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        
    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, self.min_value, self.max_value+0.9999).levels(nlevels)
        return [int(math.floor(x)) for x in ulevels]
    
    def ppf(self, x):
        return int(math.floor(self.min_value + x*(self.max_value + 0.9999 - self.min_value)))

class CategoricalUncertainty(Uncertainty):
    
    def __init__(self, name, categories):
        super(CategoricalUncertainty, self).__init__(name)
        self.categories = categories
        
    def levels(self, nlevels):
        ilevels = IntegerUncertainty(self.name, 0, len(self.categories)-1).levels(nlevels)
        return [self.categories[i] for i in ilevels]
    
    def ppf(self, x):
        return self.categories[int(math.floor(x*(len(self.categories)-0.0001)))]
    
class NamedObjectMap(object):
    
    def __init__(self, type):
        super(NamedObjectMap, self).__init__()
        self.type = type
        self._data = OrderedDict()
        
        if not issubclass(type, NamedObject):
            raise TypeError("type must be a NamedObject")
        
    def clear(self):
        self._data = OrderedDict()
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        if not isinstance(value, self.type):
            raise TypeError("can only add " + self.type.__name__ + " objects")
        
        if value.name != key:
            raise ValueError("key does not match name of " + self.type.__name__)
        
        self._data[key] = value
        
    def __delitem__(self, key):
        del self._data[key]
        
    def __iter__(self):
        return iter(self._data.values())
    
    def __contains__(self, item):
        return item in self._data
    
    def extend(self, value):
        if hasattr(value, "__iter__"):
            for item in value:
                if not isinstance(item, self.type):
                    raise TypeError("can only add " + self.type.__name__ + " objects")
                
            for item in value:
                self._data[item.name] = item
        elif isinstance(value, Parameter):
            self._data[value.name] = value
        else:
            raise TypeError("can only add " + str(type) + " objects")
            
    def __add__(self, value):
        self.extend(value)
        return self
        
    def __iadd__(self, value):
        self.extend(value)
        return self
    
    def keys(self):
        return self._data.keys()
    
    def __getattr__(self, name):
        return getattr(self._data, name)
        
class ParameterMap(NamedObjectMap):
    
    def __init__(self):
        super(ParameterMap, self).__init__(Parameter)

class ResponseMap(NamedObjectMap):
    
    def __init__(self):
        super(ResponseMap, self).__init__(Response)
        
class LeverMap(NamedObjectMap):
    
    def __init__(self):
        super(LeverMap, self).__init__(Lever)
        
class UncertaintyMap(NamedObjectMap):
    
    def __init__(self):
        super(UncertaintyMap, self).__init__(Uncertainty)

class Model(object):
    
    def __init__(self, function):
        super(Model, self).__init__()
        self.function = function
        self._parameters = ParameterMap()
        self._responses = ResponseMap()
        self.constraints = []
        self._levers = LeverMap()
        self._uncertainties = UncertaintyMap()
        self.fixed_parameters = {}
        
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self._parameters.extend(value)
        
    @property
    def responses(self):
        return self._responses
    
    @responses.setter
    def responses(self, value):
        self._responses.extend(value)
        
    @property
    def levers(self):
        return self._levers
        
    @levers.setter
    def levers(self, value):
        self._levers.extend(value)
        
    @property
    def uncertainties(self):
        return self._uncertainties
    
    @uncertainties.setter
    def uncertainties(self, value):
        self._uncertainties.extend(value)
        
    def fix(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.fixed_parameters.update(arg)
            else:
                raise RhodiumError("fix() only accepts keyword arguments or a dict")
            
        self.fixed_parameters.update(kwargs)
        
class ListOfDict(list):
    """A list of dictionary objects.
    
    Represents a list of dictionary objects but provides additional methods for
    querying and transforming the data.
    """
        
    def append(self, sample):
        if not isinstance(sample, dict):
            raise TypeError("ListOfDict can only contain dict objects")
        
        super(ListOfDict, self).append(sample)
        
    def __str__(self):
        result = ""
        
        for i in range(len(self)):
            result += "Index "
            result += str(i)
            result += ":\n"
            
            for key in self[i]:
                result += "    "
                result += str(key)
                result += ": "
                result += str(self[i][key])
                result += "\n"
                
        return result
    
    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            indices,keys = pos
            
            if isinstance(indices, slice):
                indices = xrange(*indices.indices(len(self)))
            elif isinstance(indices, int):
                indices = [indices]
                
            if not isinstance(keys, list) and not isinstance(keys, tuple):
                keys = [keys]

            result = ListOfDict()
                
            for i in indices:
                submap = {}
                    
                for key in keys:
                    submap[key] = super(ListOfDict, self).__getitem__(i)[key]
                    
                result.append(submap)
                    
            return result
        elif isinstance(pos, str):
            return self.as_list(pos)
        else:
            return super(ListOfDict, self).__getitem__(pos)
        
    def as_list(self, key=None, index=0):
        result = []
        
        if len(self) == 0:
            return result
        
        if key is None:
            if len(self[0].keys()) > 1:
                raise ValueError("Can not convert ListOfDict to list that contains more than one key")
            else:
                key = list(self[0].keys())[0]
            
        for i in range(len(self)):
            value = super(ListOfDict, self).__getitem__(i)[key]
            result.append(value[index] if isinstance(value, list) else value)
                
        return result
    
    def as_array(self, keys=None, index=0):
        import numpy
        
        if len(self) == 0:
            return numpy.empty([0])
        
        if keys is None:
            keys = self[0].keys()
            
        if isinstance(keys, str):
            keys = [keys]
            
        if isinstance(keys, set):
            keys = list(keys)
    
        if len(keys) == 1:
            key = keys[0]
            result = numpy.empty([len(self)], dtype=numpy.dtype(type(self[0][key])))
            
            for i, env in enumerate(self):
                result[i] = env[key][index] if isinstance(env[key], list) else env[key]
        else:
            dt = { "names" : keys, "formats" : [numpy.dtype(type(self[0][key])) for key in keys] }
            result = numpy.empty([len(self)], dtype=dt)
    
            for i, env in enumerate(self):
                result[i] = tuple(env[key][index] if isinstance(env[key], list) else env[key] for key in keys)
        
        return result

def _fix_generator(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        for sample in samples:
            result = sample.copy()
            result.update(fixed_parameters)
            yield result
    else:
        result = samples.copy()
        result.update(fixed_parameters)
        yield result
    
def fix(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        return _fix_generator(samples, fixed_parameters)
    else:
        result = samples.copy()
        result.update(fixed_parameters)
        return result
        
def _is_feasible(model, result):
    for constraint in model.constraints:
        if hasattr(constraint, "__call__"):
            if not constraint(result.copy()):
                return False
        else:
            if not eval(constraint.expr, result.copy()):
                return False
        
    return True

def check_feasibility(model, results):
    if isinstance(results, dict):
        return _is_feasible(model, results)
    else:
        return [_is_feasible(model, result) for result in results]
        
def mean(values):
    return float(sum(values)) / len(values)
