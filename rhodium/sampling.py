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
import random
from collections import OrderedDict
from .model import DataSet
        
def sample_uniform(model, nsamples):
    """Returns a data set with uncertainty parameters sampled using a uniform distribution."""
    result = DataSet()
        
    for i in range(nsamples):
        entry = OrderedDict()
        
        for uncertainty in model.uncertainties:
            entry[uncertainty.name] = uncertainty.ppf(random.uniform(0.0, 1.0))
    
        result.append(entry)
        
    return result
        
def sample_lhs(model, nsamples):
    """Returns a data set with uncertainty parameters sampled using Latin hypercube sampling."""
    if len(model.uncertainties) == 0:
        raise ValueError("model has no uncertainties defined")
    
    samples = OrderedDict()
    
    for uncertainty in model.uncertainties:
        levels = uncertainty.levels(nsamples)
        random.shuffle(levels)
        samples[uncertainty.name] = levels
        
    result = DataSet()
        
    for i in range(nsamples):
        entry = OrderedDict()
        
        for key, values in samples.items():
            entry[key] = values[i]
            
        # include model fixed parameters
        for key in model.fixed_parameters.keys():
            entry[key] = model.fixed_parameters[key]

        result.append(entry)
        
    return result
