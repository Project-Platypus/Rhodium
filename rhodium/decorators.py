# Copyright 2015-2024 David Hadka
#
# This file is part of Rhodium, a Python module for robust decision
# making and exploratory modeling.
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
import random
import inspect
from abc import ABCMeta
from .model import Constraint, Lever, Model, Parameter, Response, \
    Uncertainty, CategoricalLever, IntegerLever, PermutationLever, \
    SubsetLever, RealLever, UniformUncertainty, NormalUncertainty, \
    LogNormalUncertainty, Direction

class UnnamedObject(metaclass=ABCMeta):
    """Base class for model components not yet assigned a name.

    Rhodium has the concept of a `NamedObject`, which simply means a name is
    associated with an object, typically one of Rhodium's classes like `Lever`
    or `Uncertainty`.  But when using decorators to construct a model, the
    name is derived from the keyword arguments to a function, which are not
    known until we can inspect the function.  Thus, this class represents the
    `NamedObject` before it is assigned a name.

    Parameters
    ----------
    constructor : Callable
        The constructor that converts this unnamed object into a named object
    """

    def __init__(self, constructor, *args, **kwargs):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

    def convert(self, name):
        return self.constructor(name, *self.args, **self.kwargs)

class Real(UnnamedObject):
    """Construct a `RealLever` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(RealLever, *args, **kwargs)

class Integer(UnnamedObject):
    """Construct a `IntegerLever` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(IntegerLever, *args, **kwargs)

class Categorical(UnnamedObject):
    """Construct a `CategoricalLever` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(CategoricalLever, *args, **kwargs)

class Permutation(UnnamedObject):
    """Construct a `PermutationLever` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(PermutationLever, *args, **kwargs)

class Subset(UnnamedObject):
    """Construct a `SubsetLever` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(SubsetLever, *args, **kwargs)

class Uniform(float, UnnamedObject):
    """Construct a `UniformUncertainty` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(UniformUncertainty, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))

class Normal(float, UnnamedObject):
    """Construct a `NormalUncertainty` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(NormalUncertainty, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))

class LogNormal(float, UnnamedObject):
    """Construct a `LogNormalUncertainty` using the decorator approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(LogNormalUncertainty, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, kwargs.get("default_value", float("NaN")))

class Minimize(Response):

    def __init__(self, name, **kwargs):
        super().__init__(name, direction=Direction.MINIMIZE, **kwargs)

class Maximize(Response):

    def __init__(self, name, **kwargs):
        super().__init__(name, direction=Direction.MAXIMIZE, **kwargs)

class Info(Response):

    def __init__(self, name, **kwargs):
        super().__init__(name, direction=Direction.INFO, **kwargs)

class Ignore(Response):

    def __init__(self, name=None, **kwargs):
        super().__init__("Ignored" + str(random.randint()) if name is None else name, direction=Direction.IGNORE, **kwargs)

class CallableModel(Model):

    def __init__(self, function):
        super().__init__(function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class RhodiumModel:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)

        argspec = inspect.getfullargspec(f)
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

class Parameters:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)

        for v in self.args:
            if isinstance(v, str):
                f.rhodium_model.parameters[v] = Parameter(v)
            elif isinstance(v, Parameter):
                f.rhodium_model.parameters[v.name] = v
            else:
                raise ValueError("arg must be a string or Parameter")

        for k, v in self.kwargs.items():
            if isinstance(v, dict):
                f.rhodium_model.parameters[k] = Parameter(k, **v)
            elif isinstance(v, (list, tuple)):
                f.rhodium_model.parameters[k] = Parameter(k, *v)
            else:
                f.rhodium_model.parameters[k] = Parameter(k, v)

        return f

class Responses:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)

        for v in self.args:
            if isinstance(v, str):
                f.rhodium_model.responses[v] = Response(v, Response.INFO)
            elif isinstance(v, Response):
                f.rhodium_model.responses[v.name] = v
            else:
                raise ValueError("arg must be a string or Response")

        for k, v in self.kwargs.items():
            if isinstance(v, UnnamedObject):
                f.rhodium_model.responses[k] = v.convert(k)
            elif isinstance(v, dict):
                f.rhodium_model.responses[k] = Response(k, **v)
            elif isinstance(v, (list, tuple)):
                f.rhodium_model.responses[k] = Response(k, *v)
            else:
                f.rhodium_model.responses[k] = Response(k, v)

        return f

class Constraints:

    def __init__(self, *args):
        self.args = args

    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)

        for v in self.args:
            if isinstance(v, Constraint):
                f.rhodium_model.constraints.append(v)
            elif isinstance(v, str) or callable(v):
                f.rhodium_model.constraints.append(Constraint(v))
            else:
                raise ValueError("arg must be a Constraint, string, or callable")

        return f

class Levers:

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

        for k, v in self.kwargs.items():
            if isinstance(v, UnnamedObject):
                f.rhodium_model.levers[k] = v.convert(k)
            else:
                raise ValueError("kwarg must be a Lever")

        return f

class Uncertainties:

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

        for k, v in self.kwargs.items():
            if isinstance(v, UnnamedObject):
                f.rhodium_model.uncertainties[k] = v.convert(k)
            else:
                raise ValueError("kwarg must be an Uncertainty")

        return f
