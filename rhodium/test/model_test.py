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
from rhodium.model import *

class TestConstraint(unittest.TestCase):
    
    def testSimpleConstraint(self):
        c = Constraint("x < 1")
        self.assertTrue(c.is_feasible({ "x" : 0 }))
        self.assertFalse(c.is_feasible({ "x" : 1 }))
        
        self.assertEquals(0, c.distance({ "x" : 0 }))
        self.assertNotEquals(0, c.distance({ "x" : 1 }))
        
    def testComplexConstraint(self):
        c = Constraint("x < 1 and y > 1")
        self.assertTrue(c.is_feasible({ "x" : 0, "y" : 2 }))
        self.assertFalse(c.is_feasible({ "x" : 0, "y" : 1 }))
        self.assertFalse(c.is_feasible({ "x" : 1, "y" : 1 }))
        
        self.assertEquals(0, c.distance({ "x" : 0, "y" : 2 }))
        self.assertNotEquals(0, c.distance({ "x" : 0, "y" : 1 }))
        self.assertNotEquals(0, c.distance({ "x" : 1, "y" : 1 }))

class TestModelParameters(unittest.TestCase):
    
    def test(self):
        m = Model("foo")
        p = Parameter("x")
        
        m.parameters = [p]

        self.assertEquals(1, len(m.parameters))
        self.assertEquals(p, m.parameters[0])
        self.assertEquals(p, m.parameters["x"])
        self.assertTrue("x" in m.parameters)
        
    def testOrder(self):
        m = Model("foo")
        p1 = Parameter("x")
        p2 = Parameter("y")
        p3 = Parameter("z")
        
        m.parameters = [p1, p2, p3]
        
        self.assertEquals(3, len(m.parameters))
        self.assertEquals(p1, m.parameters[0])
        self.assertEquals(p2, m.parameters[1])
        self.assertEquals(p3, m.parameters[2])
        
    def testInvalidType(self):
        m = Model("foo")
        with self.assertRaises(TypeError):
            m.parameters = [Response("x")])
        
class TestUniformUncertainty(unittest.TestCase):
    
    def testLevels(self):
        uu = UniformUncertainty("x", 0.0, 1.0)
        
        levels = uu.levels(50)
        self.assertTrue(all(i >= 0.0 and i <= 1.0 for i in levels))
        
        levels = uu.levels(3)
        self.assertTrue(all(i >= 0.0 and i <= 1.0 for i in levels))
        
    def testPpf(self):
        uu = UniformUncertainty("x", 0.0, 1.0)
        
        self.assertEquals(0.0, uu.ppf(0.0))
        self.assertEquals(0.5, uu.ppf(0.5))
        self.assertEquals(1.0, uu.ppf(1.0))

class TestIntegerUncertainty(unittest.TestCase):
    
    def testLevels(self):
        iu = IntegerUncertainty("x", 0, 10)
        
        levels = iu.levels(50)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))
        
        levels = iu.levels(3)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))
        
    def testPpf(self):
        iu = IntegerUncertainty("x", 0, 10)
        
        self.assertEquals(0, iu.ppf(0.0))
        self.assertEquals(5, iu.ppf(0.5))
        self.assertEquals(10, iu.ppf(1.0))
        
class TestCategoricalUncertainty(unittest.TestCase):
    
    def testLevels(self):
        categories = ["a", "b", "c"]
        cu = CategoricalUncertainty("x", categories)
        
        levels = cu.levels(50)
        self.assertTrue(all(i in categories for i in levels))
        
        levels = cu.levels(3)
        self.assertTrue(all(i in categories for i in levels))
        
    def testPpf(self):
        categories = ["a", "b", "c"]
        cu = CategoricalUncertainty("x", categories)
        
        self.assertEquals("a", cu.ppf(0.0))
        self.assertEquals("b", cu.ppf(0.5))
        self.assertEquals("c", cu.ppf(1.0))
