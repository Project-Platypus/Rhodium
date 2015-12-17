# Copyright 2015 David Hadka
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

import six
import mpldatacursor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ColorConverter
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.mplot3d import Axes3D

class HandlerSizeLegend(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p1 = mpatches.Circle(xy=(0.2 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
                             radius=(height*0.25)/2)
        self.update_prop(p1, orig_handle, legend)
        p1.set_transform(trans)
        
        p2 = mpatches.Circle(xy=(0.66 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
                             radius=(height*1.0)/2)
        self.update_prop(p2, orig_handle, legend)
        p2.set_transform(trans)
        
        return [p1, p2]

def scatter3d(model, data,
           x = None,
           y = None,
           z = None,
           c = None,
           s = None,
           s_range = (10, 50),
           show_colorbar = True,
           show_legend = True,
           **kwargs):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(x, six.string_types):
        x_label = x
        x = [d[x_label] for d in data]
    else:
        x_label = None
            
    if isinstance(y, six.string_types):
        y_label = y
        y = [d[y_label] for d in data]
    else:
        y_label = None
        
    if isinstance(z, six.string_types):
        z_label = z
        z = [d[z_label] for d in data]
    else:
        z_label = None
        
    if isinstance(c, six.string_types):
        c_label = c
        c = [d[c_label] for d in data]
    else:
        c_label = None
        
    if isinstance(s, six.string_types):
        s_label = s
        s = [d[s_label] for d in data]
    else:
        s_label = None
        
    remaining_keys = set(model.responses.keys())
    
    used_keys = set([x_label, y_label, z_label, c_label, s_label])
    used_keys.remove(None)
    
    if used_keys.issubset(remaining_keys):
        remaining_keys -= used_keys
    else:
        remaining_keys = set()

    for key in remaining_keys:
        if x is None:
            x_label = key
            x = [d[x_label] for d in data]
        elif y is None:
            y_label = key
            y = [d[y_label] for d in data]
        elif z is None:
            z_label = key
            z = [d[z_label] for d in data]
        elif c is None:
            c_label = key
            c = [d[c_label] for d in data]
        elif s is None:
            s_label = key
            s = [d[s_label] for d in data]
        
    if z is None:
        z = 0
        
    if c is None:
        c = 'b'
        show_colorbar = False
        
    if s is None:
        s = 20
        show_legend = False
    else:
        s_min = min(s)
        s_max = max(s)
        s = (s_range[1]-s_range[0]) * ((s-s_min) / (s_max-s_min)) + s_range[0]

    handle = ax.scatter(xs = x,
                        ys = y,
                        zs = z,
                        c = c,
                        s = s,
                        **kwargs)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
        
    if show_colorbar:
        cb = fig.colorbar(handle, shrink=0.5, aspect=5)
        cb.set_label(c_label)
    
    if show_legend:
        proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
        ax.legend([proxy],
                  [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
                  handler_map={mpatches.Circle: HandlerSizeLegend()},
                  **kwargs)
        
    def formatter(**kwargs):
        i = kwargs.get("ind")[0]
        point = data[i]
        keys = model.responses.keys()
        label = "Index %d" % i
        
        for key in keys:
            label += "\n%s: %0.2f" % (key, point[key])
        
        return label
        
    mpldatacursor.datacursor(formatter=formatter, hover=True)
        
    return fig

def scatter2d(model, data,
           x = None,
           y = None,
           c = None,
           s = None,
           s_range = (10, 50),
           show_colorbar = True,
           show_legend = True,
           **kwargs):
    fig = plt.figure(facecolor='white')
    ax = plt.gca()
    
    if isinstance(x, six.string_types):
        x_label = x
        x = [d[x_label] for d in data]
    else:
        x_label = None
            
    if isinstance(y, six.string_types):
        y_label = y
        y = [d[y_label] for d in data]
    else:
        y_label = None
        
    if isinstance(c, six.string_types):
        c_label = c
        c = [d[c_label] for d in data]
    else:
        c_label = None
        
    if isinstance(s, six.string_types):
        s_label = s
        s = [d[s_label] for d in data]
    else:
        s_label = None
        
    remaining_keys = set(model.responses.keys())
    
    used_keys = set([x_label, y_label, c_label, s_label])
    used_keys.remove(None)
    
    if used_keys.issubset(remaining_keys):
        remaining_keys -= used_keys
    else:
        remaining_keys = set()

    for key in remaining_keys:
        if x is None:
            x_label = key
            x = [d[x_label] for d in data]
        elif y is None:
            y_label = key
            y = [d[y_label] for d in data]
        elif c is None:
            c_label = key
            c = [d[c_label] for d in data]
        elif s is None:
            s_label = key
            s = [d[s_label] for d in data]
        
    if c is None:
        c = 'b'
        show_colorbar = False
        
    if s is None:
        s = 20
        show_legend = False
    else:
        s_min = min(s)
        s_max = max(s)
        s = (s_range[1]-s_range[0]) * ((s-s_min) / (s_max-s_min)) + s_range[0]

    handle = plt.scatter(x = x,
                         y = y,
                         c = c,
                         s = s,
                         **kwargs)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
        
    if show_colorbar:
        cb = fig.colorbar(handle, shrink=0.5, aspect=5)
        cb.set_label(c_label)
    
    if show_legend:
        proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
        ax.legend([proxy],
                  [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
                  handler_map={mpatches.Circle: HandlerSizeLegend()},
                  **kwargs)
        
    def formatter(**kwargs):
        i = kwargs.get("ind")[0]
        point = data[i]
        keys = model.responses.keys()
        label = "Index %d" % i
        
        for key in keys:
            label += "\n%s: %0.2f" % (key, point[key])
        
        return label
        
    mpldatacursor.datacursor(formatter=formatter, hover=True)
        
    return fig

def parallel(model, data):
    
    