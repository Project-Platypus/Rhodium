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

import os
import ast
import six
import math
import inspect
import random
import operator
import pandas as pd
import scipy.stats as stats
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from platypus import Real, Integer

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
    
    def __init__(self, name, default_value = None, **kwargs):
        super(Parameter, self).__init__(name)
        self.default_value = default_value

        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)
        
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
    INFO = 2
    IGNORE = 0
    
    def __init__(self, name, type = INFO, **kwargs):
        super(Response, self).__init__(name)
        self.type = type
        
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)
        
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
        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_distance"]
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        
        if isinstance(self.expr, str):
            self._convert()
        
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
    
    @abstractmethod
    def from_variables(self, variables):
        raise NotImplementedError("method not implemented")
    
class RealLever(Lever):
    
    def __init__(self, name, min_value, max_value, length = 1):
        super(RealLever, self).__init__(name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.length = length
        
    def to_variables(self):
        return [Real(self.min_value, self.max_value) for _ in range(self.length)]
    
    def from_variables(self, variables):
        if self.length == 1:
            return variables[0]
        else:
            return variables
    
class IntegerLever(Lever):
    
    def __init__(self, name, min_value, max_value, length = 1):
        super(IntegerLever, self).__init__(name)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.length = length
        
    def to_variables(self):
        return [Integer(self.min_value, self.max_value) for _ in range(self.length)]
    
    def from_variables(self, variables):
        if self.length == 1:
            return variables[0]
        else:
            return variables
    
class CategoricalLever(Lever):
    
    def __init__(self, name, categories):
        super(CategoricalLever, self).__init__(name)
        self.categories = categories
        
    def to_variables(self):
        return [Integer(0, len(self.categories)-1)]
    
    def from_variables(self, variables):
        return self.categories[variables[0]]
        
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
        return self.min_value + x*(self.max_value - self.min_value)
    
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
        if isinstance(key, six.integer_types):
            for i, (k, v) in enumerate(six.iteritems(self._data)):
                if i == key:
                    return v
            raise KeyError(key)
        else:
            return self._data[key]
    
    def __setitem__(self, key, value):
        if not isinstance(value, self.type):
            raise TypeError("can only add " + self.type.__name__ + " objects")
        
        if isinstance(key, six.integer_types):
            self._data = OrderedDict([(value.name, value) if i==key else (k, v) for i, (k, v) in enumerate(six.iteritems(self._data))])
        else: 
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
    
    #def __getattr__(self, name):
    #    return getattr(self._data, name)
        
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
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        pass
        
    def fix(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.fixed_parameters.update(arg)
            else:
                raise RhodiumError("fix() only accepts keyword arguments or a dict")
            
        self.fixed_parameters.update(kwargs)
        
class DataSet(list):

    def __init__(self, data=[]):
        super(DataSet, self).__init__()
        
        if isinstance(data, six.string_types):
            self.load(data)
        else:
            for entry in data:
                self.append(entry)
        
    def append(self, sample):
        if not isinstance(sample, dict):
            raise TypeError("DataSet can only contain dict objects")
        
        super(DataSet, self).append(sample)
        
    def __str__(self):
        result = ""
        
        if len(self) == 0:
            result += "Empty (no feasible solutions)"
        
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
    
    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))
    
    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            indices,keys = pos
            
            if isinstance(indices, slice):
                indices = xrange(*indices.indices(len(self)))
            elif isinstance(indices, int):
                indices = [indices]
                
            if not isinstance(keys, list) and not isinstance(keys, tuple):
                keys = [keys]

            result = DataSet()
                
            for i in indices:
                submap = {}
                    
                for key in keys:
                    submap[key] = super(DataSet, self).__getitem__(i)[key]
                    
                result.append(submap)
                    
            return result
        elif isinstance(pos, str):
            return self.as_list(pos)
        elif isinstance(pos, slice):
            indices = xrange(*pos.indices(len(self)))
            result = DataSet()
            
            for i in indices:
                result.append(super(DataSet, self).__getitem__(i))
                
            return result    
        else:
            return super(DataSet, self).__getitem__(pos)
        
    def __setitem__(self, pos, value):
        if isinstance(pos, str):
            if isinstance(value, (list, tuple)) and len(value) == len(self):
                for i, o in enumerate(self):
                    o[pos] = value[i]
            else:
                for o in self:
                    o[pos] = value
        else:
            return super(DataSet, self).__setitem__(pos)
        
    def _trim(self, value, index=None):
        if index is not None and isinstance(value, (list, tuple)):
            return value[index]
        else:
            return value
        
    def as_list(self, key=None, index=None):
        result = []
        
        if len(self) == 0:
            return result
        
        if key is None:
            if len(self[0].keys()) > 1:
                raise ValueError("Can not convert DataSet to list that contains more than one key")
            else:
                key = list(self[0].keys())[0]
            
        for i in range(len(self)):
            value = super(DataSet, self).__getitem__(i)[key]
            result.append(self._trim(value, index))
                
        return result
    
    def as_dataframe(self, keys=None, index=None):
        dict = OrderedDict()
        
        if keys is None:
            keys = self[0].keys()
            
        if isinstance(keys, str):
            keys = [keys]
    
        for key in keys:
            dict[key] = [self._trim(d[key], index) for d in self]
            
        return pd.DataFrame(dict)
    
    def as_array(self, keys=None, index=None):
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
                result[i] = self._trim(env[key], index)
        else:
            dt = { "names" : keys, "formats" : [numpy.dtype(type(self[0][key])) for key in keys] }
            result = numpy.empty([len(self)], dtype=dt)
    
            for i, env in enumerate(self):
                result[i] = tuple(self._trim(env[key], index) for key in keys)
        
        return result
    
    def find(self, expr):
        result = DataSet()
        
        for entry, cond in zip(self, self.apply(expr)):
            if cond:
                result.append(entry)
                    
        return result
                
    def apply(self, expr):
        result = []
        result_keys = []
        
        # parse the expression to determine any simple result names
        if isinstance(expr, str):
            root = ast.parse(expr, mode="exec")
            
            if isinstance(root.body[0], ast.Assign):
                target = root.body[0].targets[0]
                
                if isinstance(target, ast.Tuple):
                    result_keys.extend([t.id for t in target.elts if isinstance(t, ast.Name)])
                elif isinstance(target, ast.Name):
                    result_keys.append(target.id)
                    
        for entry in self:
            tmp_env = {}
            tmp_env.update(_eval_env)
            tmp_env.update(entry)
            
            if isinstance(expr, str):
                if len(result_keys) > 0:
                    six.exec_(expr, {}, tmp_env)
                    
                    for key in result_keys:
                        entry[key] = tmp_env[key]
                
                    if len(result_keys) == 1:
                        result.append(tmp_env[key])
                    else:
                        result.append([tmp_env[key] for key in result_keys])
                else:
                    result.append(eval(expr, {}, tmp_env))
            else:
                result.append(expr(tmp_env))
                
        return result
          
    def find_min(self, key):
        index, _ = min(enumerate([d[key] for d in self]), key=operator.itemgetter(1))
        return self[index]
    
    def find_max(self, key):
        index, _ = max(enumerate([d[key] for d in self]), key=operator.itemgetter(1))
        return self[index]
    
    def save(self, file, format=None, **kwargs):
        if format is None:
            _, format = os.path.splitext(file)
            
            if len(format) > 0 and format[0] == ".":
                format = format[1:]
                
        if format == "xls" or format == "xlsx":
            self.as_dataframe().to_excel(file, **kwargs)
        elif format == "csv":
            self.as_dataframe().to_csv(file, **kwargs)
        elif format == "json":
            self.as_dataframe().to_json(file, **kwargs)
        elif format == "pkl":
            self.as_dataframe().to_pickle(file, **kwargs)
        else:
            raise ValueError("unsupported file format '%s'" % str(format))
        
    def load(self, file, format=None, **kwargs):
        if format is None:
            _, format = os.path.splitext(file)
            
            if len(format) > 0 and format[0] == ".":
                format = format[1:]
            
        if format == "xls" or format == "xlsx":
            df = pd.read_excel(file, **kwargs)
        elif format == "csv":
            df = pd.read_csv(file, **kwargs)
        elif format == "json":
            df = pd.read_json(file, **kwargs)
        elif format == "pkl":
            df = pd.read_pickle(file, **kwargs)
        else:
            raise ValueError("unsupported file format '%s'" % str(format))
        
        for i, row in df.iterrows():
            entry = OrderedDict()
                
            for key in df.columns.values:
                entry[key] = row[key]
                
            self.append(entry)

def _overwrite_generator(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        for sample in samples:
            result = sample.copy()
            result.update(fixed_parameters)
            yield result
    else:
        result = samples.copy()
        result.update(fixed_parameters)
        yield result
    
def overwrite(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        return _overwrite_generator(samples, fixed_parameters)
    else:
        result = samples.copy()
        result.update(fixed_parameters)
        return result
    
def _update_generator(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        for sample in samples:
            result = fixed_parameters.copy()
            result.update(sample)
            yield result
    else:
        result = fixed_parameters.copy()
        result.update(samples)
        yield result
    
def update(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        return _update_generator(samples, fixed_parameters)
    else:
        result = fixed_parameters.copy()
        result.update(samples)
        return result
    
def populate_defaults(model, samples):
    if isinstance(samples, dict):
        samples = [samples]
        
    argspec = inspect.getargspec(model.function)
    default_values = {k:v for k, v in zip(argspec.args[-len(argspec.defaults):], argspec.defaults)}
    
    for sample in samples:
        for parameter in model.parameters:
            if parameter.name not in sample:
                if parameter.default_value is not None:
                    sample[parameter.name] = parameter.default_value
                else:
                    sample[parameter.name] = default_values[parameter.name]
                    
