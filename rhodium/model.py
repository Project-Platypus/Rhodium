import ast
import math
import inspect
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from platypus.experimenter import Job, submit_jobs
from platypus.types import Real
from platypus.core import Problem

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
        
        return eval(self.expr, {}, tmp_env)
    
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
        
class Lever(object):
    """Defines an adjustable lever that controls a model parameter.
    
    Model parameters can either be constant, controlled by a lever, or
    subject to uncertainty.  The lever defines the available options for
    a given design factor.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Lever, self).__init__()
        
    @abstractmethod
    def to_variables(self):
        raise NotImplementedError("method not implemented")
    
class RealLever(Lever):
    
    def __init__(self, min_value, max_value, length = 1):
        super(RealLever, self).__init__()
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.length = length
        
    def to_variables(self):
        return [Real(self.min_value, self.max_value) for _ in range(self.length)]
        
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
        self._args = inspect.getargspec(model.function).args
        
    def run(self):
        args = {}
        
        for parameter in self.model.parameters:
            if parameter.name in self.sample:
                args[parameter.name] = self.sample[parameter.name]
            elif parameter.default_value:
                args[parameter.name] = parameter.default_value
                
        raw_output = self.model.function(**args)
        
        # support output as a dict or list-like object
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

def _to_problem(model):
    variables = []
    
    for name, lever in model.levers.iteritems():
        variables.extend(lever.to_variables())
    
    nvars = len(variables)
    nobjs = sum([1 if r.type == Response.MINIMIZE or r.type == Response.MAXIMIZE else 0 for r in model.responses])
    nconstrs = len(model.constraints)
    
    def function(vars):
        env = {}
        offset = 0
        
        for name, lever in model.levers.iteritems():
            env[name] = vars[offset:offset+lever.length]
            offset += lever.length
            
        job = EvaluateJob(model, env)
        job.run()
        
        objectives = [job.output[r.name] for r in model.responses]
        constraints = [constraint.distance(job.output) for constraint in model.constraints]
        
        return objectives, constraints
    
    problem = Problem(nvars, nobjs, nconstrs, function)
    problem.types[:] = variables
    problem.directions[:] = [Problem.MINIMIZE if r.type == Response.MINIMIZE else Problem.MAXIMIZE for r in model.responses if r.type == Response.MINIMIZE or r.type == Response.MAXIMIZE]
    problem.constraints[:] = "==0"
    return problem

def optimize(model, algorithm="NSGAII", NFE=10000, **kwargs):
    module = __import__("platypus.algorithms", fromlist=[''])
    class_ref = getattr(module, algorithm)
    
    args = kwargs.copy()
    args["problem"] = _to_problem(model)
    
    instance = class_ref(**args)
    instance.run(NFE)
    
    result = []
    
    for solution in instance.result:
        env = {}
        offset = 0
        
        for name, lever in model.levers.iteritems():
            if lever.length == 1:
                env[name] = solution.variables[offset]
            else:
                env[name] = solution.variables[offset:offset+lever.length]

            offset += lever.length
        
        for i, response in enumerate(model.responses):
            env[response.name] = solution.objectives[i]
            
        result.append(env)
        
    return result
        
        
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
