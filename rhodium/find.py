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

import operator
from .model import _eval_env, ListOfDict

def find(data, expr):
    result = ListOfDict()
    
    for entry in data:
        tmp_env = {}
        tmp_env.update(_eval_env)
        tmp_env.update(entry)
        
        if isinstance(expr, str):
            if eval(expr, {}, tmp_env):
                result.append(entry)
        else:
            if expr(tmp_env):
                result.append(entry)
                
    return result
                
def which(data, expr):
    result = []
    
    for entry in data:
        tmp_env = {}
        tmp_env.update(_eval_env)
        tmp_env.update(entry)
        
        if isinstance(expr, str):
            result.append(eval(expr, {}, tmp_env))
        else:
            result.append(expr(tmp_env)) 
            
    return result
          
def find_min(data, key):
    index, _ = min(enumerate([d[key] for d in data]), key=operator.itemgetter(1))
    return data[index]

def find_max(data, key):
    index, _ = max(enumerate([d[key] for d in data]), key=operator.itemgetter(1))
    return data[index]
                