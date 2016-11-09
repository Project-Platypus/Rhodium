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
from ..ffi import *

class TestNativeModel(unittest.TestCase):
    
#     @classmethod
#     def setUpClass(cls):
#         from distutils.ccompiler import new_compiler, show_compilers
#         
#         # determine the relative path to the source file
#         dir = os.path.dirname(__file__)
#         cwd = os.getcwd()
#         common_prefix = os.path.commonprefix([dir, cwd])
#         rel_dir = os.path.relpath(dir, cwd)
#         src = os.path.join(rel_dir, "test.c")
#         
#         compiler = new_compiler("mingw32")
#         (obj,) = compiler.compile([src]) #, extra_preargs=["-m64"])
#         compiler.link_shared_lib([obj], "test")
#         
#         libname = compiler.library_filename("test", lib_type="shared")
#         cls.sopath = os.path.join(rel_dir, libname)

    def testNormalReturn(self):
        model = NativeModel("test.dll", "norm_return")
        model.parameters = [Parameter("x", type="double"),
                            Parameter("y", type="double")]
        model.responses = [Response("z", type="double")]
        result = evaluate(model, {"x" : 3, "y" : 5})
        self.assertEquals(15, result["z"])
        
    def testArgumentReturn(self):
        model = NativeModel("test.dll", "arg_return")
        model.parameters = [Parameter("x", type="double"),
                            Parameter("y", type="double")]
        model.responses = [Response("z", type="double", asarg=True)]
        result = evaluate(model, {"x" : 3, "y" : 5})
        self.assertEquals(15, result["z"])

    def testSum(self):
        model = NativeModel("test.dll", "sum")
        model.parameters = [Parameter("x", type="double*10")]
        model.responses = [Response("sum", type="double")]
        result = evaluate(model, {"x" : [1, 2, 3, 4, 5]})
        self.assertEquals(15, result["sum"])
        
    def testArrayAdd(self):
        model = NativeModel("test.dll", "array_add")
        model.parameters = [Parameter("x", type="double*", len_arg="n"),
                            Parameter("y", type="double*", len_arg="n"),
                            Parameter("n", type="int")]
        model.responses = [Response("z", type="double*", len_arg="n", asarg=True)]
        result = evaluate(model, {"x" : [1, 2, 3, 4, 5], "y" : [5, 4, 3, 2, 1]})
        self.assertEquals([6, 6, 6, 6, 6], result["z"])
 