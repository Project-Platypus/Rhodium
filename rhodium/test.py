import six
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

class Uniform(UnnamedObject):
    
    def __init__(self, *args, **kwargs):
        super(Uniform, self).__init__(UniformUncertainty, *args, **kwargs)
        
class Minimize(UnnamedObject):
    
    def __init__(self, **kwargs):
        super(Minimize, self).__init__(Response, type=Response.MINIMIZE, **kwargs)
        
class Maximize(UnnamedObject):
    
    def __init__(self, **kwargs):
        super(Maximize, self).__init__(Response, type=Response.MAXIMIZE, **kwargs)
     
class Info(UnnamedObject):
    
    def __init__(self, **kwargs):
        super(Info, self).__init__(Response, type=Response.INFO, **kwargs)
        
class Ignore(UnnamedObject):
    
    def __init__(self, **kwargs):
        super(Ignore, self).__init__(Response, type=Response.IGNORE, **kwargs)

class CallableModel(Model):
    
    def __init__(self, function):
        super(CallableModel, self).__init__(function)
    
    def __call__(self, *vars, **kwargs):
        return self.function(*vars, **kwargs)

class RhodiumModel(object):
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, f):
        if not hasattr(f, "rhodium_model"):
            f.rhodium_model = CallableModel(f)
            
        for k in inspect.getargspec(f).args:
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
    
    
@RhodiumModel()
@Responses(x=Minimize(eps=0.01), y=Maximize(eps=0.05))
@Levers(x=Real(-10, 10))
@Uncertainties(p=Uniform(0.0, 1.0))
def test(x):
    return x**2, (x-2)**2


# print(test(5, 10))
# 
# output = evaluate(test, [{"x" : 5, "y" : 10},
#                          {"x" : 1, "y" : 7},
#                          {"x" : 2, "y" : 1}])
# 
# scatter3d(test, output, x="x", y="y", z="z")
# plt.show()

output = optimize(test, "NSGAII", 10000)
print(output)

scatter2d(test, output)
parallel_coordinates(test, output, colormap="rainbow")
plt.show()