from .model import *

class OpenMDAOModel(Model):
    
    def __init__(self, problem):
        super(OpenMDAOModel, self).__init__(self._evaluate)

        from openmdao.api import Problem
        if not isinstance(problem, Problem):
            raise ValueError("problem must be an OpenMDAO Problem instance")

        self.problem = problem
        
    def _evaluate(self, **kwargs):
        result = {}
        
        for parameter in self.parameters:
            key = parameter.name
            
            if hasattr(parameter, "connect"):
                key = getattr(parameter, "connect")
                
            self.problem.root.unknowns[key] = kwargs[parameter.name]
            
        self.problem.run()
            
        for response in self.responses:
            key = response.name
            
            if hasattr(response, "connect"):
                key = getattr(response, "connect")
                
            result[response.name] = self.problem.root.unknowns[key]
            
        return result