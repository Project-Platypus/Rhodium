import math
import inspect
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pyDOE import lhs
from platypus.experimenter import Job, submit_jobs

class RhodiumError(Exception):
    pass

class Parameter(object):
    
    def __init__(self, name, default_value = None):
        super(Parameter, self).__init__()
        self.name = name
        self.default_value = default_value
        
class Uncertainty(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Uncertainty, self).__init__()
        
    @abstractmethod
    def cast(self, value):
        """Converts a value in [0, 1] to a value defined by this uncertainty."""
        raise NotImplementedError("method not implemented")
        
class RealUncertainty(Uncertainty):
    
    def __init__(self, min_value, max_value):
        super(RealUncertainty, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def cast(self, value):
        return value*(self.max_value - self.min_value) + self.min_value
    
class IntegerUncertainty(Uncertainty):
    
    def __init__(self, min_value, max_value):
        super(IntegerUncertainty, self).__init__()
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        
    def cast(self, value):
        return math.floor(value*(self.max_value - self.min_value + 0.9999) + self.min_value)

class CategoricalUncertainty(Uncertainty):
    
    def __init__(self, categories):
        super(CategoricalUncertainty, self).__init__()
        self.categories = categories
        
    def cast(self, value):
        index = math.floor(value*(len(self.categories) + 0.9999))
        return self.categories[index]
    
class Response(object):
    
    def __init__(self, name):
        super(Response, self).__init__()
        self.name = name

class Model(object):
    
    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name
        self.parameters = []
        self.uncertainties = {}
        self.responses = []
        self.function = None
        self.fixed_parameters = {}
        
    def fix(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.fixed_parameters.update(arg)
            else:
                raise RhodiumError("fix() only accepts keyword arguments or a dict")
            
        self.fixed_parameters.update(kwargs)
        
def sample_lhs(model, nsamples, **kwargs):
    uncertain_parameters = list(model.uncertainties.keys())
    samples = lhs(len(uncertain_parameters), nsamples, **kwargs)
    
    for i in range(samples.shape[0]):
        entry = []
        
        for j in range(len(model.parameters)):
            name = model.parameters[j].name
            
            if name in uncertain_parameters:
                entry.append(model.uncertainties[name].cast(samples[i][j]))
            elif model.parameters[j].default_value is not None:
                entry.append(model.parameters[j].default_value)
            else:
                raise RhodiumError("parameter " + name + " is not uncertain and has no default value")
    
        yield entry
        
def generate_jobs(model, samples):
    for sample in samples:
        yield EvaluateJob(model, sample)

class EvaluateJob(Job):
    
    def __init__(self, model, sample):
        super(EvaluateJob, self).__init__()
        self.model = model
        self.sample = sample
        
    def run(self):
        kwargs = {}
            
        for i in range(len(model.parameters)):
            kwargs[model.parameters[i].name] = self.sample[i]
                
        self.output = self.model.function(**kwargs)

def evaluate(model, samples, **kwargs):
    results = submit_jobs(generate_jobs(model, samples), **kwargs)
    return [result.output for result in results]
        
def rosen(x, y):
    return (1 - x)**2 + 100*(y-x**2)**2


    

           
model = Model("Rosenbrock")
model.parameters = [Parameter("x"), Parameter("y")]
model.responses = [Response("f(x,y)")]
model.uncertainties = {"x" : RealUncertainty(-10.0, 10.0),
                       "y" : RealUncertainty(-10.0, 10.0)}
model.function = rosen

model.fix({"x" : 0.25, "y" : 0.75})

samples = list(sample_lhs(model, 1000))
results = evaluate(model, samples)

print samples
print results

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
