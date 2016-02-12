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
from ..model import *
from ..optimization import *
from ..sampling import *
from ..excel import *

class TestExcelHelper(unittest.TestCase):

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testGetItem(self):
        file = os.path.join(os.path.dirname(__file__), "TestGetItem.xlsx")
        
        with ExcelHelper(file) as helper:
            self.assertEqual(1, helper["A1"])
            self.assertEqual(5, helper["A5"])
            self.assertEqual(((1.0,),(2.0,),(3.0,),(4.0,),(5.0,)), helper["A1:A5"])
            self.assertEqual(None, helper["B2"])
            self.assertEqual(u"hello", helper["C2"])
            self.assertEqual(((u"hello", u"world",),), helper["C2:D2"])
            
            helper.set_sheet(2)
            self.assertEqual(u"sheet 2", helper["B2"])
            
    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testSetItem(self):
        file = os.path.join(os.path.dirname(__file__), "TestSetItem.xlsx")
        
        with ExcelHelper(file) as helper:
            helper["A1"] = 1
            self.assertEqual(6, helper["A3"])
            
            helper["A2"] = 3
            self.assertEqual(4, helper["A3"])
            
            helper["A1:A2"] = [6, 7]
            self.assertEqual(13, helper["A3"])
            
            helper.set_sheet(2)
            helper["B2"] = "hello"
            helper.set_sheet(1)
            helper["B2"] = "world"
            helper.set_sheet(2)
            self.assertEqual(u"hello", helper["B2"])

class TestExcelModel(unittest.TestCase):
    
    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testEvaluate(self):
        file = os.path.join(os.path.dirname(__file__), "TestModel.xlsx")
        
        with ExcelModel(file) as model:
            model.parameters = [Parameter("X1", cell="B1"),
                                Parameter("X2", cell="B2")]
            model.responses = [Response("Y", cell="B3")]
  
            result = evaluate(model, {"X1" : 3, "X2" : 5})
            self.assertEqual(8, result["Y"])
        
    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")    
    def testSample(self):
        file = os.path.join(os.path.dirname(__file__), "TestModel.xlsx")
        
        with ExcelModel(file) as model:
            model.parameters = [Parameter("X1", cell="B1"),
                                Parameter("X2", cell="B2")]
            model.responses = [Response("Y", cell="B3")]
            model.uncertainties = [UniformUncertainty("X1", 0.0, 1.0),
                                   IntegerUncertainty("X2", 2, 5)]
  
            samples = sample_lhs(model, 100)
            results = evaluate(model, samples)
            
            for result in results:
                self.assertEqual(result["Y"], result["X1"] + result["X2"])
            