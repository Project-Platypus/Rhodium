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
import ast

_eval_env = {}
_modules = ["math"]

for modulename in _modules:
    module = __import__(modulename, fromlist=[''])
    for name in dir(module):
        if not name.startswith("_"):
            _eval_env[name] = getattr(module, name)

def _has_assignment(tree):
    return isinstance(tree.body[0], ast.Assign)

def _get_result_keys(tree):
    keys = []
    
    if _has_assignment(tree):
        target = tree.body[0].targets[0]
                
        if isinstance(target, ast.Tuple):
            keys.extend([t.id for t in target.elts if isinstance(t, ast.Name)])
        elif isinstance(target, ast.Name):
            keys.append(target.id)

    return keys

def _evaluate(expr, env, update_env=True):
    tmp_env = {}
    tmp_env.update(_eval_env)
    tmp_env.update(env)
            
    if isinstance(expr, six.string_types):
        tree = ast.parse(expr, mode="exec")
        result_keys = _get_result_keys(tree)
        
        if update_env and len(result_keys) > 0:
            six.exec_(expr, {}, tmp_env)
                    
            for key in result_keys:
                env[key] = tmp_env[key]
                
            if len(result_keys) == 1:
                return tmp_env[result_keys[0]]
            else:
                return [tmp_env[key] for key in result_keys]
        else:
            return eval(expr, {}, tmp_env)
    elif six.callable(expr):
        tmp_env = {}
        tmp_env.update(env)
        
        return expr(tmp_env)
    else:
        raise ValueError("expr must be a string or a callable")
    
def _evaluate_all(expr, iterable, update_env=True):
    result = []
            
    if isinstance(expr, six.string_types):
        tree = ast.parse(expr, mode="exec")
        result_keys = _get_result_keys(tree)
            
        for env in iterable:
            tmp_env = {}
            tmp_env.update(_eval_env)
            tmp_env.update(env)
            
            if update_env and len(result_keys) > 0:
                six.exec_(expr, {}, tmp_env)
                        
                for key in result_keys:
                    env[key] = tmp_env[key]
                    
                if len(result_keys) == 1:
                    result.append(tmp_env[result_keys[0]])
                else:
                    result.append([tmp_env[key] for key in result_keys])
            else:
                result.append(eval(expr, {}, tmp_env))
    elif six.callable(expr):
        for env in iterable:
            tmp_env = {}
            tmp_env.update(env)
            
            result.append(expr(tmp_env))
    else:
        raise ValueError("expr must be a string or a callable")
    
    return result
