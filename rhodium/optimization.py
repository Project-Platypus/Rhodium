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
from collections import OrderedDict
from platypus import Job, Problem, unique
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
            elif parameter.default_value:
                args[parameter.name] = parameter.default_value
                
        # evaluate the model
        raw_output = self.model.function(**args)
        
        # support output as a dict or list-like object
        if isinstance(raw_output, dict):
            args.update(raw_output)
        else:
            for i, response in enumerate(self.model.responses):
                if response.type is not Response.IGNORE:
                    args[response.name] = raw_output[i]
            
        self.output = args

def evaluate(model, samples, evaluator=None):
    if evaluator is None:
        from .config import RhodiumConfig
        
        evaluator = RhodiumConfig.default_evaluator
    
    if inspect.isgenerator(samples) or (hasattr(samples, '__iter__') and not isinstance(samples, dict)):
        results = evaluator.evaluate_all(generate_jobs(model, samples))
        return DataSet([result.output for result in results])
    else:
        results = evaluator.evaluate_all(generate_jobs(model, samples))
        return results[0].output

def _to_problem(model):
    variables = []
    lengths = []
    
    for lever in model.levers:
        vars = lever.to_variables()
        variables.extend(vars)
        lengths.append(len(vars))
    
    nvars = len(variables)
    nobjs = sum([1 if r.type == Response.MINIMIZE or r.type == Response.MAXIMIZE else 0 for r in model.responses])
    nconstrs = len(model.constraints)
    
    def function(vars):
        env = {}
        offset = 0
        
        for i, lever in enumerate(model.levers):
            env[lever.name] = lever.from_variables(vars[offset:(offset+lengths[i])])
            offset += lengths[i]
            
        job = EvaluateJob(model, env)
        job.run()
        
        objectives = [job.output[r.name] for r in model.responses if r.type in [Response.MINIMIZE, Response.MAXIMIZE]]
        constraints = [constraint.distance(job.output) for constraint in model.constraints]
        
        if nconstrs > 0:
            return objectives, constraints
        else:
            return objectives
    
    problem = Problem(nvars, nobjs, nconstrs, function)
    problem.types[:] = variables
    problem.directions[:] = [Problem.MINIMIZE if r.type == Response.MINIMIZE else Problem.MAXIMIZE for r in model.responses if r.type == Response.MINIMIZE or r.type == Response.MAXIMIZE]
    problem.constraints[:] = "==0"
    return problem

def optimize(model, algorithm="NSGAII", NFE=10000, **kwargs):
    module = __import__("platypus", fromlist=[''])
    class_ref = getattr(module, algorithm)
    
    args = kwargs.copy()
    args["problem"] = _to_problem(model)
    
    instance = class_ref(**args)
    instance.run(NFE)
    
    result = DataSet()
    
    for solution in unique(instance.result):
        if not solution.feasible:
            continue
        
        env = OrderedDict()
        offset = 0
        
        for lever in model.levers:
            if lever.length == 1:
                env[lever.name] = solution.variables[offset]
            else:
                env[lever.name] = solution.variables[offset:offset+lever.length]

            offset += lever.length
        
        if any([r.type not in [Response.MINIMIZE, Response.MAXIMIZE] for r in model.responses]):
            # if there are any responses not included in the optimization, we must
            # re-evaluate the model to get all responses
            env = evaluate(model, env)
        else:
            for i, response in enumerate(model.responses):
                env[response.name] = solution.objectives[i]
            
        result.append(env)
        
    return result