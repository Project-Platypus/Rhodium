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
import os
import sys
import unittest
import pandas as pd
from ..brush import *

class TestBrush(unittest.TestCase):
    
    def testSimpleBrush(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        brush = Brush("col1 < 2")
        assignment = apply_brush(brush, df)
        self.assertEqual(assignment[0], brush.name)
        self.assertEqual(assignment[1], RhodiumConfig.default_unassigned_label)
        
    def testComplexBrush(self):
        d = {'col1': [1, 2, 3], 'col2': [3, 4, 5]}
        df = pd.DataFrame(data=d)
        brush = Brush("col1 <= 2 and col2 >= 4")
        assignment = apply_brush(brush, df)
        self.assertEqual(assignment[0], RhodiumConfig.default_unassigned_label)
        self.assertEqual(assignment[1], brush.name)
        self.assertEqual(assignment[2], RhodiumConfig.default_unassigned_label)
        
    def testMultipleBrush(self):
        d = {'col1': [1, 2, 3], 'col2': [3, 4, 5]}
        df = pd.DataFrame(data=d)
        brush1 = Brush("col1 < 2")
        brush2 = Brush("col2 > 4")
        assignment = apply_brush(BrushSet([brush1, brush2]), df)
        self.assertEqual(assignment[0], brush1.name)
        self.assertEqual(assignment[1], RhodiumConfig.default_unassigned_label)
        self.assertEqual(assignment[2], brush2.name)
        
