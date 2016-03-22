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
from collections import OrderedDict

class Brush(object):
    
    def __init__(self, name, expr, color=None):
        super(Brush, self).__init__()
        self.name = name
        self.expr = expr
        self.color = color
        
class BrushMap(object):
    
    def __init__(self, brushes):
        super(BrushMap, self).__init__()
        self.map = OrderedDict()
        
        if not isinstance(brushes, (list, tuple)):
            brushes = [brushes]
        
        for brush in brushes:
            if isinstance(brush, Brush):
                self.map[brush.name] = brush
            elif isinstance(brush, BrushMap):
                self.map.update(brush.map)
            elif isinstance(brush, six.string_types):
                self.map[brush] = Brush(brush, brush)
            else:
                raise ValueError("only Brush or string expressions can be added to BrushMap")
                
    def __len__(self):
        return len(self.map)
    
    def keys(self):
        return six.iterkeys(self.map)
                
    def __getitem__(self, key):
        return self.map[key]
    
    def __iter__(self):
        return six.itervalues(self.map)
    
    def colors(self):
        return [brush.color for brush in self]