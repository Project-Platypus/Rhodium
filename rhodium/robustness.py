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

import numpy as np
import scipy.spatial as sp
from .model import *
from .sampling import *
from .optimization import *

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

def regret_type1(model, results, baseline, percentile=90):
    quantiles = []
    
    for response in model.responses:
        if response.dir == Response.MINIMIZE or response.dir == Response.MAXIMIZE:
            values = [abs((result[response.name] - baseline[response.name]) / baseline[response.name]) for result in results]
            quantiles.append(np.percentile(values, percentile))
            
    return max(quantiles)

def regret_type2(model, all_results, baseline_results, percentile=90):
    # for each uncertainty sampling, find the best value
    best = []
    
    for i in range(len(all_results[0])):
        entry = {}
        
        for response in model.responses:
            if response.dir == Response.MINIMIZE:
                entry[response.name] = min([result[i][response.name] for result in all_results])
            elif response.dir == Response.MAXIMIZE:
                entry[response.name] = max([result[i][response.name] for result in all_results])
                
        best.append(entry)
    
    # then compute the regret from the best value
    quantiles = []
    
    for response in model.responses:
        if response.dir == Response.MINIMIZE or response.dir == Response.MAXIMIZE:
            values = []
            
            for i in range(len(all_results[0])):
                values.append(abs((baseline_results[i][response.name] - best[i][response.name]) / baseline_results[i][response.name]))
            
            quantiles.append(np.percentile(values, percentile))
        
    return max(quantiles)

def satisficing_type1(model, results, expr=None):
    if expr is None:
        return mean(check_feasibility(model, results))
    else:
        return [expr(result) for result in results]
    
def satisficing_type2(model, results, baseline, expr=None):
    distances = []
    
    # ensure all default parameters are defined in baseline
    baseline = baseline.copy()
    populate_defaults(model, baseline)
    
    for result in results:
        if (expr is None and _is_feasible(model, result)) or (expr is not None and expr(result)):
            distances.append(sp.distance.euclidean(
                    [result[u.name] for u in model.uncertainties],
                    [baseline[u.name] for u in model.uncertainties]))
            
    return 0.0 if len(distances) == 0 else min(distances)

def evaluate_robustness(model, policies, SOWs=1000, in_place=True, return_all=False):
    if isinstance(SOWs, six.integer_types):
        SOWs = sample_lhs(model, SOWs)
    
    # evaluate the policies
    n = len(policies)
    all_results = [evaluate(model, update(SOWs, policy)) for policy in policies]
    
    # compute the standard robustness metrics
    result = policies if in_place else DataSet()
    
    result["Regret Type 1"] = [regret_type1(model, all_results[i], policies[i]) for i in range(n)]
    result["Regret Type 2"] = [regret_type2(model, all_results, all_results[i]) for i in range(n)]
    result["Satisficing Type 1"] = [satisficing_type1(model, all_results[i]) for i in range(n)]
    result["Satisficing Type 2"] = [satisficing_type2(model, all_results[i], policies[i]) for i in range(len(policies))]
    
    if return_all:
        return SOWs, result, all_results
    else:
        return result