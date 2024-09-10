# Copyright 2015-2024 David Hadka
#
# This file is part of Rhodium, a Python module for robust decision
# making and exploratory modeling.
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
from ..model import Direction
from ..decorators import Minimize, Maximize, Info, Ignore

class TestResponses(unittest.TestCase):

    def testMinimize(self):
        r = Minimize("foo")
        self.assertEqual("foo", r.name)
        self.assertEqual(Direction.MINIMIZE, r.direction)
        
    def testMaximize(self):
        r = Maximize("foo")
        self.assertEqual("foo", r.name)
        self.assertEqual(Direction.MAXIMIZE, r.direction)
        
    def testInfo(self):
        r = Info("foo")
        self.assertEqual("foo", r.name)
        self.assertEqual(Direction.INFO, r.direction)
        
    def testIgnore(self):
        r = Ignore("foo")
        self.assertEqual("foo", r.name)
        self.assertEqual(Direction.IGNORE, r.direction)
