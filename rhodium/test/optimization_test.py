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

def schaffer(x):
    return [x**2, (x-2)**2]

class TestOptimization(unittest.TestCase):
    
    def testReal(self):   
        model = Model(schaffer)
        
        model.parameters = [Parameter("x")]
        model.responses = [Response("f1", Response.MINIMIZE),
                           Response("f2", Response.MINIMIZE)]
        model.levers = [RealLever("x", -10.0, 10.0)]

        output = optimize(model, "NSGAII", 10000)
        
        self.assertTrue(len(output) > 0)
        
        for i in range(len(output)):
            self.assertTrue("x" in output[i])
            self.assertTrue("f1" in output[i])
            self.assertTrue("f2" in output[i])
        
    def testInteger(self):
        model = Model(schaffer)
        
        model.parameters = [Parameter("x")]
        model.responses = [Response("f1", Response.MINIMIZE),
                           Response("f2", Response.MINIMIZE)]
        model.levers = [IntegerLever("x", -10, 10)]

        output = optimize(model, "NSGAII", 10000)
        
        self.assertTrue(len(output) > 0)
        
        for i in range(len(output)):
            self.assertTrue("x" in output[i])
            self.assertTrue(isinstance(output[i]["x"], int))
            self.assertTrue("f1" in output[i])
            self.assertTrue("f2" in output[i])
