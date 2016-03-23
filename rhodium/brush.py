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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from pandas.core.frame import DataFrame
from rhodium.config import RhodiumConfig

class Brush(object):
    
    def __init__(self, name, expr=None, color=None):
        super(Brush, self).__init__()
        
        if expr is None:
            expr = name
        
        self.name = name
        self.expr = expr
        self.color = color
        
class BrushSet(object):
    
    def __init__(self, brushes):
        super(BrushSet, self).__init__()
        self.map = OrderedDict()
        
        if not isinstance(brushes, (list, tuple)):
            brushes = [brushes]
        
        for brush in brushes:
            if isinstance(brush, Brush):
                self.map[brush.name] = brush
            elif isinstance(brush, BrushSet):
                self.map.update(brush.map)
            elif isinstance(brush, six.string_types):
                self.map[brush] = Brush(brush, brush)
            else:
                raise ValueError("only Brush or string expressions can be added to BrushSet")
                
    def __len__(self):
        return len(self.map)
    
    def keys(self):
        return six.iterkeys(self.map)
    
    def __contains__(self, key):
        return key in self.map
                
    def __getitem__(self, key):
        return self.map[key]
    
    def __iter__(self):
        return six.itervalues(self.map)
    
    def colors(self):
        return [brush.color for brush in self]
    
def apply_brush(brush, data):
    if isinstance(data, DataFrame):
        return _apply_brush_dataframe(brush, data)
    else:
        raise ValueError("unsupported type, data must be a DataFrame")
        
def _apply_brush_dataframe(brush_set, df):
    assignment = [None]*df.shape[0];
        
    for brush in brush_set:
        bin = df.query(brush.expr)
        
        for i in bin.index:
            if assignment[i] is None:
                assignment[i] = brush.name
                
    for i, a in enumerate(assignment):
        if a is None:
            assignment[i] = RhodiumConfig.default_unassigned_label
            
    return assignment

def brush_color_map(brush, assignment):
    cc = mpl.colors.colorConverter
    color_map = OrderedDict()

    if RhodiumConfig.default_unassigned_label in assignment:
        color_map[RhodiumConfig.default_unassigned_label] = RhodiumConfig.default_unassigned_brush_color
    
    for a in assignment:
        if a not in color_map:
            if a in brush:
                if brush[a].color is None:
                    color_map[a] = None
                else:
                    color_map[a] = cc.to_rgb(brush[a].color)
            else:
                color_map[a] = None
                
    unassigned_count = sum([1 if x is None else 0 for x in six.itervalues(color_map)])
    
    if unassigned_count > 0:
        brush_colors = RhodiumConfig.default_brush_colors
        count = 0
        
        for k, v in six.iteritems(color_map):
            if v is None:
                color_map[k] = brush_colors[count]
                count += 1
                        
    return color_map

def color_brush(brush, data, **kwargs):
    assignment = apply_brush(brush, data)
    color_map = brush_color_map(brush, assignment)                
    return ([color_map[a] for a in assignment], color_map)
