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

import os
import six
import pandas as pd
from .model import Parameter, Response, DataSet, Model

class FileModel(Model):
    
    def __init__(self):
        super(FileModel, self).__init__(self._evaluate)
        
    def _evaluate(self, **kwargs):
        raise NotImplementedError("models loaded from files do not support evaluation")

def load(file, format=None, parameters=[], **kwargs):
    if format is None:
        _, format = os.path.splitext(file)
            
        if len(format) > 0 and format[0] == ".":
            format = format[1:]
                
    if format == "xls" or format == "xlsx":
        df = pd.read_excel(file, **kwargs)
    elif format == "csv":
        df = pd.read_csv(file, **kwargs)
    elif format == "json":
        df = pd.read_json(file, **kwargs)
    elif format == "pkl":
        df = pd.read_pickle(file, **kwargs)
    else:
        raise ValueError("unsupported file format '%s'" % str(format))
    
    names = list(df.columns.values)
    data = DataSet()
    
    if isinstance(parameters, six.string_types):
        parameters = [parameters]
    
    for i in range(df.shape[0]):
        entry = {}
        
        for j in range(df.shape[1]):
            entry[names[j]] = df.iloc[i,j]
            
        data.append(entry)
        
    model = FileModel()
    model.parameters = [Parameter(names[j] if isinstance(j, six.integer_types) else j) for j in parameters]
    model.responses = [Response(names[j]) for j in range(df.shape[1]) if j not in parameters and names[j] not in parameters]
        
    return (model, data)