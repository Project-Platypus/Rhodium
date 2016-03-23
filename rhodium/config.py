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
from __future__ import absolute_import, division, print_function

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from platypus.config import PlatypusConfig

class _RhodiumConfig(object):
    
    def __init__(self):
        super(_RhodiumConfig, self).__init__()
        self.default_cmap = plt.get_cmap("rainbow")
        self.default_brush_colors = sns.color_palette()
        self.default_unassigned_brush_color = mpl.colors.colorConverter.to_rgb("#CCCCCC")
        self.default_unassigned_label = "Unassigned"
     
    @property
    def default_evaluator(self):
        return PlatypusConfig.default_evaluator
     
    @default_evaluator.setter
    def default_evaluator(self, value):
        PlatypusConfig.default_evaluator = value
        
    @property
    def default_log_frequency(self):
        return PlatypusConfig.default_log_frequency
    
    @default_log_frequency.setter
    def default_log_frequency(self, value):
        PlatypusConfig.default_log_frequency = value
        
RhodiumConfig = _RhodiumConfig()
