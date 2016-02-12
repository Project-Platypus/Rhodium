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
import unittest
from rhodium.model import IntegerUncertainty

class TestIntegerUncertainty(unittest.TestCase):
    
    def testLevels(self):
        iu = IntegerUncertainty("x", 0, 10)
        
        levels = iu.levels(50)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))
        
        levels = iu.levels(3)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))