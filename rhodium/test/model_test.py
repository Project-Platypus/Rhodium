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
import numpy as np
from rhodium.model import IntegerUncertainty
from rhodium.model import TriangularUncertainty
from rhodium.model import PointUncertainty


class TestIntegerUncertainty(unittest.TestCase):
    
    def testLevels(self):
        iu = IntegerUncertainty("x", 0, 10)
        
        levels = iu.levels(50)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))
        
        levels = iu.levels(3)
        self.assertTrue(all(i >= 0 and i <= 10 for i in levels))
        self.assertTrue(all(isinstance(i, six.integer_types) for i in levels))


class TestTriangularUncertainty(unittest.TestCase):

    def test_ppf_0_1_symmetric(self):
        """Check the PPF on a symmetric triangular distribution on (0, 1)."""
        tu = TriangularUncertainty('x', 0, 1, 0.5)

        self.assertEqual(tu.ppf(0), 0)
        self.assertEqual(tu.ppf(1), 1)
        self.assertEqual(tu.ppf(0.5), 0.5)

    def test_ppf_0_1_skew(self):
        """Check the PPF on a skewed triangular distribution on (0, 1)."""
        tu = TriangularUncertainty('x', 0, 1, 0.25)

        self.assertEqual(tu.ppf(0), 0)
        self.assertEqual(tu.ppf(1), 1)
        self.assertEqual(tu.ppf(0.25), 0.25)

    def test_ppf_10_20_skew(self):
        """Check the PPF on a skewed triangular distribution on (10, 20)."""
        tu = TriangularUncertainty('x', 10, 20, 12.5)

        self.assertEqual(tu.ppf(0), 10)
        self.assertEqual(tu.ppf(1), 20)
        self.assertEqual(tu.ppf(0.25), 12.5)

    def test_out_of_order(self):
        """Check that an error is raised if the min, max and mode are not properly ordered."""
        with self.assertRaises(ValueError):
            tu = TriangularUncertainty('x', 2, 1, 0)
        with self.assertRaises(ValueError):
            tu = TriangularUncertainty('x', 0, 1, 2)
        with self.assertRaises(ValueError):
            tu = TriangularUncertainty('x', 2, 0, 1)


class TestPointUncertainty(unittest.TestCase):
    """Unit tests for PointUncertainty"""

    def test_ppf(self):
        """Check that the ppf returns the point value for any quantile."""
        value = 1
        pu = PointUncertainty('x', value)
        for quantile in np.linspace(0, 1):
            self.assertEqual(pu.ppf(quantile), value)


if __name__ == '__main__':
    unittest.main()
