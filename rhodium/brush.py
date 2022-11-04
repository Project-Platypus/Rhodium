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
import matplotlib as mpl
from collections import OrderedDict
from pandas.core.frame import DataFrame
from .config import RhodiumConfig
from .model import DataSet
from .expr import _evaluate

class Brush(object):
    """Defines a brush to color data points matching an expression.
    
    A brush is a mechanism to highlight, or color, data that matches a given
    expression.  The expression is a string that contains one or more boolean
    or comparison operators.  Any parameter in the data set can be referenced.
    For example, if the data set contains parameters "x" and "y", valid 
    expressions include:
        x < 5
        y > 10
        x <= 5 and y > 10
    """
    
    def __init__(self, name, expr=None, color=None):
        super(Brush, self).__init__()
        
        if expr is None:
            expr = name
        
        self.name = name
        self.expr = expr
        self.color = color
        
class BrushSet(object):
    """A collection of Brush objects.
    
    If two or more brushes overlap, then the first brush, in the order they
    are defiend in this collection, determines the coloring.  For example, for
        BrushSet([Brush("x < 5", "blue"), Brush("x > 3", "green")])
    a point with value x = 4 will be colored "blue".
    """
    
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
            elif isinstance(brush, str):
                self.map[brush] = Brush(brush, brush)
            else:
                raise ValueError("only Brush or string expressions can be added to BrushSet")
                
    def __len__(self):
        return len(self.map)
    
    def keys(self):
        return self.map.keys()
    
    def __contains__(self, key):
        return key in self.map
                
    def __getitem__(self, key):
        return self.map[key]
    
    def __iter__(self):
        return iter(self.map.values())
    
    def colors(self):
        return [brush.color for brush in self]
    
def _as_brush_set(input):
    """Converts the input into a BrushSet.
    
    The input can be either a single brush or multiple brush, each defined
    as either a string (e.g., "x < 5") or a brush (e.g., Brush("x < 5")).
    """
    if isinstance(input, BrushSet):
        return input
    else:
        return BrushSet(input)
    
def apply_brush(brush_set, data):
    """Applies the given brush set to the given data, returning the color assignments."""
    if isinstance(data, DataFrame):
        return _apply_brush_dataframe(brush_set, data)
    elif isinstance(data, DataSet):
        return _apply_brush_dataset(brush_set, data)
    else:
        raise ValueError("unsupported type, data must be a DataFrame or DataSet")
    
def _apply_brush_dataset(brush_set, ds):
    brush_set = _as_brush_set(brush_set)
    n = len(ds)
    assignment = [None]*n;
        
    for brush in brush_set:
        bin = ds.apply(brush.expr)
        
        for i in range(n):
            if bin[i]:
                assignment[i] = brush.name
                  
    for i, a in enumerate(assignment):
        if a is None:
            assignment[i] = RhodiumConfig.default_unassigned_label
            
    return assignment
        
def _apply_brush_dataframe(brush_set, df):
    brush_set = _as_brush_set(brush_set)
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

def brush_color_map(brush_set, assignment):
    brush_set = _as_brush_set(brush_set)
    cc = mpl.colors.colorConverter
    color_map = OrderedDict()

    if RhodiumConfig.default_unassigned_label in assignment:
        color_map[RhodiumConfig.default_unassigned_label] = RhodiumConfig.default_unassigned_brush_color
        
    # we want to retain the original ordering of the brushes, so first determine
    # which brushes are applicable to a data set
    classes = set([a for a in assignment])
    
    for b in brush_set:
        if b.name in classes:
            color_map[b.name] = cc.to_rgba(b.color) if b.color is not None else None
             
    # determine if any brushes have no assigned color and pick one   
    unassigned_count = sum([1 if x is None else 0 for x in color_map.values()])
    
    if unassigned_count > 0:
        brush_colors = RhodiumConfig.default_brush_colors
        count = 0
        
        for k, v in color_map.items():
            if v is None:
                color_map[k] = brush_colors[count]
                count += 1
                        
    return color_map

def color_brush(brush_set, data, **kwargs):
    assignment = apply_brush(brush_set, data)
    color_map = brush_color_map(brush_set, assignment)                
    return ([color_map[a] for a in assignment], color_map)

def color_indices(c, color_map):
    values = list(color_map.values())
    return [values.index(v) for v in c]
