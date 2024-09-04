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
from .model import Model

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
