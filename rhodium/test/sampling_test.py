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
import unittest
from rhodium import *

class TestSampling(unittest.TestCase):
    
    def testUniform(self):   
        model = Model("foo")
        model.uncertainties = [UniformUncertainty("x", 5.0, 10.0)]
        
        samples = sample_uniform(model, 100)
        
        self.assertEqual(100, len(samples))

        for i in range(len(samples)):
            self.assertTrue("x" in samples[i])
            self.assertTrue(samples[i]["x"] >= 5.0 and samples[i]["x"] <= 10.0)
        
    def testLHS(self):
        model = Model("foo")
        model.uncertainties = [UniformUncertainty("x", 5.0, 10.0)]
        
        samples = sample_lhs(model, 100)
        
        self.assertEqual(100, len(samples))

        for i in range(len(samples)):
            self.assertTrue("x" in samples[i])
            self.assertTrue(samples[i]["x"] >= 5.0 and samples[i]["x"] <= 10.0)
