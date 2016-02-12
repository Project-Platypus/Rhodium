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

import unittest
from ..model import *
from ..optimization import *
from ..sampling import *
from ..openmdao import *

class TestOpenMDAOModel(unittest.TestCase):

    def test(self):
        try:
            from openmdao.api import IndepVarComp, Component, Problem, Group
        except ImportError:
            self.skipTest("OpenMDAO not available")

        # the following example is taken from OpenMDAMO's Paraboloid tutorial
        class Paraboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """
        
            def __init__(self):
                super(Paraboloid, self).__init__()
        
                self.add_param('x', val=0.0)
                self.add_param('y', val=0.0)
        
                self.add_output('f_xy', val=0.0)
        
            def solve_nonlinear(self, params, unknowns, resids):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Optimal solution (minimum): x = 6.6667; y = -7.3333
                """
        
                x = params['x']
                y = params['y']
        
                unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        
            def linearize(self, params, unknowns, resids):
                """ Jacobian for our paraboloid."""
        
                x = params['x']
                y = params['y']
                J = {}
        
                J['f_xy', 'x'] = 2.0*x - 6.0 + y
                J['f_xy', 'y'] = 2.0*y + 8.0 + x
                return J
        
        top = Problem()
        root = top.root = Group()
        
        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())
        
        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')
        
        top.setup()
        
        # here is where Rhodium begins
        model = OpenMDAOModel(top)
        model.parameters = [Parameter("x", connect="p1.x"),
                            Parameter("y", connect="p2.y")]
        model.responses = [Response("f_xy", connect="p.f_xy")]
        
        output = evaluate(model, {"x" : 3.0, "y" : -4.0})
        self.assertEqual(-15.0, output["f_xy"])
        