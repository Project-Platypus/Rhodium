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
__version__ = "1.4.0"

# Note: The following submodules are not exported here to allow for optional
# dependencies:
#   - rhodium.excel
#   - rhodium.ffi
#   - rhodium.openmdao
#   - rhodium.rbridge

from .config import RhodiumConfig

from .model import RhodiumError, Parameter, Direction, Response, Constraint, \
    Lever, RealLever, IntegerLever, CategoricalLever, PermutationLever, \
    SubsetLever, Uncertainty, UniformUncertainty, TriangularUncertainty, \
    PointUncertainty, NormalUncertainty, LogNormalUncertainty, \
    IntegerUncertainty, CategoricalUncertainty, Model, DataSet, save, load, \
    overwrite, update, populate_defaults
    
from .decorators import Real, Integer, Categorical, Permutation, Subset, \
    Uniform, Normal, LogNormal, Minimize, Maximize, Info, Ignore, \
    RhodiumModel, Parameters, Responses, Constraints, Levers, Uncertainties

from .plot import scatter2d, scatter3d, joint, pairs, kdeplot, hist, \
    interact, contour2d, contour3d, animate3d, parallel_coordinates

from .sa import SAResult, sa, oat, regional_sa

from .classification import Cart

from .cache import setup_cache, cached, cache, clear_cache

from .optimization import evaluate, optimize, robust_optimize

from .sampling import sample_uniform, sample_lhs

from .robustness import evaluate_robustness
 
from .brush import Brush, BrushSet, apply_brush, brush_color_map, \
    color_brush, color_indices

from .utils import promptToRun

from prim import Prim
from platypus import MapEvaluator, SubmitEvaluator, ApplyEvaluator, \
    PoolEvaluator, MultiprocessingEvaluator, ProcessPoolEvaluator
