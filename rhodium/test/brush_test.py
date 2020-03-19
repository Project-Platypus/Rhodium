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
import sys
import unittest
import pandas as pd
from ..brush import *

class TestBrush(unittest.TestCase):
    
    def testDataFrame(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        brush = Brush("col1 < 2")
        print(apply_brush(brush, df))
