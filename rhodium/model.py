import math
import inspect
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from platypus.experimenter import Job, submit_jobs

class RhodiumError(Exception):
    pass

class Parameter(object):
    """Defines a model parameter (i.e., input).
    
    Defines a model input (i.e., input) and an optional default value.  The
    name must be identical to the keyword argument to the method defining the
    model.
    """ 
    
    def __init__(self, name, default_value = None):
        super(Parameter, self).__init__()
        self.name = name
        self.default_value = default_value
        
class Response(object):
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
        super(Response, self).__init__()
        self.name = name
        self.type = type
        
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
        
class Lever(object):
    """Defines an adjustable lever that controls a model parameter.
    
    Model parameters can either be constant, controlled by a lever, or
    subject to uncertainty.  The lever defines the available options for
    a given design factor.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Lever, self).__init__()
    
class RealLever(Lever):
    
    def __init__(self, min_value, max_value, length = 1):
        super(RealLever, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.length = length
        
class Uncertainty(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Uncertainty, self).__init__()
        
    @abstractmethod    
    def levels(self, nlevels):
        raise NotImplementedError("method not implemented")
        
class RealUncertainty(Uncertainty):
    
    def __init__(self, min_value, max_value):
        super(RealUncertainty, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def levels(self, nlevels):
        d = (self.max_value - self.min_value) / nlevels
        result = []
        
        for i in range(nlevels):
            result.append(self.min_value + random.uniform(i*d, (i+1)*d))
        
        return result

class IntegerUncertainty(Uncertainty):
    
    def __init__(self, min_value, max_value):
        super(IntegerUncertainty, self).__init__()
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        
    def levels(self, nlevels):
        rlevels = RealUncertainty(self.min_value, self.max_value+0.9999).levels(nlevels)
        return [int(math.floor(x)) for x in rlevels]

class CategoricalUncertainty(Uncertainty):
    
    def __init__(self, categories):
        super(CategoricalUncertainty, self).__init__()
        self.categories = categories
        
    def levels(self, nlevels):
        ilevels = IntegerUncertainty(0, len(self.categories)-1).levels(nlevels)
        return [self.categories[i] for i in ilevels]

class Model(object):
    
    def __init__(self, function):
        super(Model, self).__init__()
        self.function = function
        self.parameters = []
        self.responses = []
        self.constraints = []
        self.levers = {}
        self.uncertainties = {}
        self.fixed_parameters = {}
        
    def fix(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.fixed_parameters.update(arg)
            else:
                raise RhodiumError("fix() only accepts keyword arguments or a dict")
            
        self.fixed_parameters.update(kwargs)

def _sample_lhs(model, nsamples):
    samples = {}
    
    for key, uncertainty in model.uncertainties.iteritems():
        levels = uncertainty.levels(nsamples)
        random.shuffle(levels)
        samples[key] = levels
        
    for i in range(nsamples):
        result = {}
        
        for key, values in samples.iteritems():
            result[key] = values[i]
    
        yield result
        
def sample_lhs(model, nsamples):
    return list(_sample_lhs(model, nsamples))
        
def fix(samples, fixed_parameters):
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        for sample in samples:
            result = sample.copy()
            result.update(fixed_parameters)
            yield result
    else:
        result = samples.copy()
        result.update(fixed_parameters)
        yield result
        
def generate_jobs(model, samples):
    if isinstance(samples, dict):
        yield EvaluateJob(model, samples)
    else:
        for sample in samples:
            yield EvaluateJob(model, sample)

class EvaluateJob(Job):
    
    def __init__(self, model, sample):
        super(EvaluateJob, self).__init__()
        self.model = model
        self.sample = sample
        
    def run(self):
        input = self.sample.copy()
        
        for parameter in self.model.parameters:
            if parameter.name not in input and parameter.default_value:
                input[parameter.name] = parameter.default_value
                
        raw_output = self.model.function(**self.sample)
        
        if isinstance(raw_output, dict):
            input.update(raw_output)
        else:
            for i, response in enumerate(self.model.responses):
                input[response.name] = raw_output[i]
            
        self.output = input

def evaluate(model, samples, **kwargs):
    results = submit_jobs(generate_jobs(model, samples), **kwargs)
    return [result.output for result in results]

def _is_feasible(model, result):
    for constraint in model.constraints:
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

        
# def rosen(x, y):
#     return (1 - x)**2 + 100*(y-x**2)**2


    

           
# model = Model("Rosenbrock")
# model.parameters = [Parameter("x"), Parameter("y")]
# model.responses = [Response("f(x,y)")]
# model.uncertainties = {"x" : RealUncertainty(-10.0, 10.0),
#                        "y" : RealUncertainty(-10.0, 10.0)}
# model.function = rosen
# 
# model.fix({"x" : 0.25, "y" : 0.75})
# 
# samples = list(sample_lhs(model, 1000))
# results = evaluate(model, samples)
# 
# print samples
# print results

# from sklearn import tree
# clf = tree.DecisionTreeRegressor(min_samples_leaf=int(0.05*len(samples)))
# clf.fit(samples, results)
# 
# from sklearn.externals.six import StringIO  
# import pydot 
# dot_data = StringIO() 
# tree.export_graphviz(clf, out_file=dot_data) 
# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# graph.write_pdf("tree.pdf") 
