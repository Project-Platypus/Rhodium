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

import inspect
import functools
from collections import OrderedDict
from platypus import Job, Problem, unique, nondominated
from .model import Response, DataSet

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
        # populate model arguments
        args = {}
        
        for parameter in self.model.parameters:
            if parameter.name in self.sample:
                args[parameter.name] = self.sample[parameter.name]
            elif parameter.default_value is not None:
                args[parameter.name] = parameter.default_value
                
        # evaluate the model
        raw_output = self.model.function(**args)
        
        # support output as a dict or list-like object
        if isinstance(raw_output, dict):
            args.update(raw_output)
        elif isinstance(raw_output, (list, tuple)):
            offset = 0
            
            for response in self.model.responses:
                if response.dir is not Response.IGNORE:
                    args[response.name] = raw_output[offset]
                    offset += 1
        elif len(self.model.responses) == 1 and self.model.responses[0].dir != Response.IGNORE:
            args[self.model.responses[0].name] = raw_output
            
        self.output = args

def evaluate(model, samples, evaluator=None, log_frequency=None):
    if evaluator is None:
        from .config import RhodiumConfig
        evaluator = RhodiumConfig.default_evaluator
        
    if log_frequency is None:
        from .config import RhodiumConfig
        log_frequency = RhodiumConfig.default_log_frequency
    
    if (inspect.isgenerator(samples) or (hasattr(samples, '__iter__')) and not isinstance(samples, dict)):
        results = evaluator.evaluate_all(generate_jobs(model, samples), job_name="Evaluate", log_frequency=log_frequency)
        return DataSet([result.output for result in results])
    else:
        results = evaluator.evaluate_all(generate_jobs(model, samples), job_name="Evaluate", log_frequency=log_frequency)
        return results[0].output

def _evaluation_function(vars, model, nvars, nobjs, nconstrs, levers):
    env = {}
    offset = 0
    
    for lever, length in levers:
        env[lever.name] = lever.from_variables(vars[offset:(offset+length)])
        offset += length
        
    job = EvaluateJob(model, env)
    job.run()
    
    objectives = [job.output[r.name] for r in model.responses if r.dir in [Response.MINIMIZE, Response.MAXIMIZE]]
    constraints = [constraint.distance(job.output) for constraint in model.constraints]
    
    if nconstrs > 0:
        return objectives, constraints
    else:
        return objectives

def _to_problem(model):
    variables = []
    levers = []
    
    for lever in model.levers:
        vars = lever.to_variables()
        variables.extend(vars)
        levers.append((lever, len(vars)))
    
    nvars = len(variables)
    nobjs = sum([1 if r.dir == Response.MINIMIZE or r.dir == Response.MAXIMIZE else 0 for r in model.responses])
    nconstrs = len(model.constraints)
    
    function = functools.partial(_evaluation_function,
                                 model=model,
                                 nvars=nvars,
                                 nobjs=nobjs,
                                 nconstrs=nconstrs,
                                 levers=levers)
    
    problem = Problem(nvars, nobjs, nconstrs, function)
    problem.types[:] = variables
    problem.directions[:] = [Problem.MINIMIZE if r.dir == Response.MINIMIZE else Problem.MAXIMIZE for r in model.responses if r.dir == Response.MINIMIZE or r.dir == Response.MAXIMIZE]
    problem.constraints[:] = "==0"
    return (problem, levers)

def optimize(model, algorithm="NSGAII", NFE=10000, module="platypus", **kwargs):
    module = __import__(module, fromlist=[''])
    class_ref = getattr(module, algorithm)
    
    args = kwargs.copy()
    args["problem"], levers = _to_problem(model)
    
    instance = class_ref(**args)
    instance.run(NFE)
    
    result = DataSet()
    
    print("here")
    
    for solution in unique(nondominated(instance.result)):
        if not solution.feasible:
            continue
        
        env = OrderedDict()
        offset = 0
        
        # decode from Platypus' internal representation (this should be fixed in Platypus instead)
        vars = [solution.problem.types[i].decode(solution.variables[i]) for i in range(solution.problem.nvars)]
        
        for lever, length in levers:
            env[lever.name] = lever.from_variables(vars[offset:(offset+length)])
            offset += length
        
        if any([r.dir not in [Response.MINIMIZE, Response.MAXIMIZE] for r in model.responses]):
            # if there are any responses not included in the optimization, we must
            # re-evaluate the model to get all responses
            print("reeval")
            env = evaluate(model, env)
        else:
            for i, response in enumerate(model.responses):
                env[response.name] = solution.objectives[i]
            
        result.append(env)
        
    return result

def _robust_evaluation_function(vars, model, SOWs, nvars, nobjs, nconstrs, levers, obj_aggregate, constr_aggregate):
    objectives = {}
    constraints = {}
    
    for response in model.responses:
        if response.dir in [Response.MINIMIZE, Response.MAXIMIZE]:
            objectives[response] = []
            
    for constraint in model.constraints:
        constraints[constraint] = [] 
    
    for SOW in SOWs:
        env = SOW.copy()
        offset = 0
        
        for lever, length in levers:
            env[lever.name] = lever.from_variables(vars[offset:(offset+length)])
            offset += length
            
        job = EvaluateJob(model, env)
        job.run()
        
        for response in model.responses:
            if response.dir in [Response.MINIMIZE, Response.MAXIMIZE]:
                objectives[response].append(job.output[response.name])
                
        for constraint in model.constraints:
            constraints[constraint].append(constraint.distance(job.output))
        
    if isinstance(obj_aggregate, dict):
        objective_values = [obj_aggregate[r.name](objectives[r]) for r in model.responses if r.dir in [Response.MINIMIZE, Response.MAXIMIZE]]
    else:
        objective_values = [obj_aggregate(objectives[r]) for r in model.responses if r.dir in [Response.MINIMIZE, Response.MAXIMIZE]]
        
    if isinstance(constr_aggregate, dict):
        constraint_values = [constr_aggregate[c](constraints[c]) for c in model.constraints]
    else:
        constraint_values = [constr_aggregate(constraints[c]) for c in model.constraints]

    if nconstrs > 0:
        return objective_values, constraint_values
    else:
        return objective_values

def _to_robust_problem(model, SOWs, obj_aggregate, constr_aggregate):
    variables = []
    levers = []
    
    for lever in model.levers:
        vars = lever.to_variables()
        variables.extend(vars)
        levers.append((lever, len(vars)))
    
    nvars = len(variables)
    nobjs = sum([1 if r.dir == Response.MINIMIZE or r.dir == Response.MAXIMIZE else 0 for r in model.responses])
    nconstrs = len(model.constraints)
    
    function = functools.partial(_robust_evaluation_function,
                                 model=model,
                                 SOWs=SOWs,
                                 nvars=nvars,
                                 nobjs=nobjs,
                                 nconstrs=nconstrs,
                                 levers=levers,
                                 obj_aggregate=obj_aggregate,
                                 constr_aggregate=constr_aggregate)
    
    problem = Problem(nvars, nobjs, nconstrs, function)
    problem.types[:] = variables
    problem.directions[:] = [Problem.MINIMIZE if r.dir == Response.MINIMIZE else Problem.MAXIMIZE for r in model.responses if r.dir == Response.MINIMIZE or r.dir == Response.MAXIMIZE]
    problem.constraints[:] = "==0"
    return (problem, levers)

def robust_optimize(model, SOWs, algorithm="NSGAII", NFE=10000, obj_aggregate=None, constr_aggregate=None, **kwargs):
    module = __import__("platypus", fromlist=[''])
    class_ref = getattr(module, algorithm)
    
    if obj_aggregate is None:
        from .robustness import mean
        obj_aggregate = mean
             
    if constr_aggregate is None:
        constr_aggregate = max
    
    args = kwargs.copy()
    args["problem"], levers = _to_robust_problem(model, SOWs, obj_aggregate, constr_aggregate)
    
    instance = class_ref(**args)
    instance.run(NFE)
    
    result = DataSet()
    
    for solution in unique(nondominated(instance.result)):
        if not solution.feasible:
            continue
        
        env = OrderedDict()
        offset = 0
        
        # decode from Platypus' internal representation (this should be fixed in Platypus instead)
        vars = [solution.problem.types[i].decode(solution.variables[i]) for i in range(solution.problem.nvars)]
        
        for lever, length in levers:
            env[lever.name] = lever.from_variables(vars[offset:(offset+length)])
            offset += length
        
        if any([r.dir not in [Response.MINIMIZE, Response.MAXIMIZE] for r in model.responses]):
            # if there are any responses not included in the optimization, we must
            # re-evaluate the model to get all responses
            env = evaluate(model, env)
            
        # here we copy over the objectives from the evaluated solution, which has been aggregated over all SOWs
        for i, response in enumerate([r for r in model.responses if r.dir in [Response.MINIMIZE, Response.MAXIMIZE]]):
            env[response.name] = solution.objectives[i]
            
        result.append(env)
        
    return result