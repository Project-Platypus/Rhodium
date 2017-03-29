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

import six
from pyper import *
from .model import *
import numpy as np

class RModel(Model):
    
    def __init__(self, file, function, **kwargs):
        super(RModel, self).__init__(self._evaluate)
        self.file = file
        self.r_function = function
        self.r = R(**kwargs)
        
        with open(file) as f:
            self.r(f.read())
        
    def _evaluate(self, **kwargs):
        prefix = "...rhodium."
        assigned_parameters = []
        
        for parameter in self.parameters:
            if parameter.name in kwargs:
                self.r.assign(prefix + parameter.name, kwargs[parameter.name])
                assigned_parameters.append(parameter)
        
        self.r(prefix + "..result = " + self.r_function + "(" + ",".join([prefix + p.name for p in assigned_parameters]) + ")")
        r_result = self.r[prefix + "..result"]
        
        result = {}
        
        if isinstance(r_result, (list, tuple, np.ndarray)):
            for i, response in enumerate(self.responses):
                result[response.name] = r_result[i]
        elif isinstance(r_result, dict):
            for response in self.responses:
                result[response.name] = r_result[response.name]
        else:
            if len(self.responses) > 1:
                raise ValueError("received more than one response from R")
            else:
                result[self.responses[0].name] = r_result
        
        return result