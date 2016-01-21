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
import warnings
import mpldatacursor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib.colors import ColorConverter
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.mplot3d import Axes3D

class HandlerSizeLegend(HandlerPatch):
    def __call__(self, legend, orig_handle,
             fontsize,
             handlebox):
        print("Here")
    
    def create_artists(self, legend, orig_handle,
                      xdescent, ydescent, width, height, fontsize, trans):
        print("Create Artists")
        p1 = mpatches.Circle(xy=(0.2 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
                             radius=(height*0.25)/2)
        self.update_prop(p1, orig_handle, legend)
        p1.set_transform(trans)
        
        p2 = mpatches.Circle(xy=(0.66 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
                             radius=(height*1.0)/2)
        self.update_prop(p2, orig_handle, legend)
        p2.set_transform(trans)
        
        return [p1, p2]
    
def to_dataframe(model, data, keys = None):
    dict = {}
    
    if keys is None:
        keys = model.responses.keys()

    for key in keys:
        dict[key] = [d[key] for d in data]
        
    return pd.DataFrame(dict)

def scatter3d(model, data,
           x = None,
           y = None,
           z = None,
           c = None,
           s = None,
           s_range = (10, 50),
           show_colorbar = True,
           show_legend = False,
           interactive = False,
           expr = None,
           class_label = "class",
           **kwargs):
    df = to_dataframe(model, data)
    
    if "axes.facecolor" in mpl.rcParams:
        orig_facecolor = mpl.rcParams["axes.facecolor"]
        mpl.rcParams["axes.facecolor"] = "white"
        
    if expr is not None:
        df[class_label] = [0.0]*df.shape[0]
        
        if isinstance(expr, six.string_types):
            expr = [expr]
            
        for i, e in enumerate(expr):
            bin = df.query(e)
            df.loc[bin.index, class_label] = float(i+1) / len(expr)
            
        if c is not None:
            warnings.warn("color and expr arguments both assigned, discarding color", UserWarning)
            
        c = class_label
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(x, six.string_types):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, six.string_types):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(z, six.string_types):
        z_label = z
        z = df[z_label]
    else:
        z_label = None
        
    if isinstance(c, six.string_types):
        c_label = c
        c = df[c_label]
    else:
        c_label = None
        
    if isinstance(s, six.string_types):
        s_label = s
        s = df[s_label]
    else:
        s_label = None
        
    used_keys = set([x_label, y_label, z_label, c_label, s_label])
    used_keys.remove(None)
    
    remaining_keys = set(model.responses.keys())
    remaining_keys -= used_keys

    for key in remaining_keys:
        if x is None:
            x_label = key
            x = df[x_label]
        elif y is None:
            y_label = key
            y = df[y_label]
        elif z is None:
            z_label = key
            z = df[z_label]
        elif c is None:
            c_label = key
            c = df[c_label]
        elif s is None:
            s_label = key
            s = df[s_label]
        
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
        if expr is None:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5)
            cb.set_label(c_label)
        else:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5,
                              ticks=[(i+0.5)/(len(expr)+1) for i in range(len(expr)+1)])
            plt.clim(0, 1)
            cb.set_label(c_label)
            cb.ax.set_xticklabels(["Unassigned"] + expr)
            cb.ax.set_yticklabels(["Unassigned"] + expr)
    
#     if show_legend:
#         proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
#         ax.legend([proxy],
#                   [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
#                   handler_map={proxy: HandlerSizeLegend()})
        
    if interactive:
        def formatter(**kwargs):
            i = kwargs.get("ind")[0]
            point = data[i]
            keys = model.responses.keys()
            label = "Index %d" % i
            
            for key in keys:
                label += "\n%s: %0.2f" % (key, point[key])
            
            return label
            
        mpldatacursor.datacursor(formatter=formatter, hover=True)
    
    if "axes.facecolor" in mpl.rcParams:
        mpl.rcParams["axes.facecolor"] = orig_facecolor
        
    return fig

def scatter2d(model, data,
           x = None,
           y = None,
           c = None,
           s = None,
           s_range = (10, 50),
           show_colorbar = True,
           show_legend = False,
           interactive = False,
           expr = None,
           class_label = "class",
           **kwargs):
    df = to_dataframe(model, data)
    fig = plt.figure(facecolor='white')
    ax = plt.gca()
    
    if expr is not None:
        df[class_label] = [0.0]*df.shape[0]
        
        if isinstance(expr, six.string_types):
            expr = [expr]
            
        for i, e in enumerate(expr):
            bin = df.query(e)
            df.loc[bin.index, class_label] = float(i+1) / len(expr)
            
        if c is not None:
            warnings.warn("color and expr arguments both assigned, discarding color", UserWarning)
            
        c = class_label
    
    if isinstance(x, six.string_types):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, six.string_types):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(c, six.string_types):
        c_label = c
        c = df[c_label]
    else:
        c_label = None
        
    if isinstance(s, six.string_types):
        s_label = s
        s = df[s_label]
    else:
        s_label = None
        
    used_keys = set([x_label, y_label, c_label, s_label])
    used_keys.remove(None)
    
    remaining_keys = set(model.responses.keys())
    remaining_keys -= used_keys

    for key in remaining_keys:
        if x is None:
            x_label = key
            x = df[x_label]
        elif y is None:
            y_label = key
            y = df[y_label]
        elif c is None:
            c_label = key
            c = df[c_label]
        elif s is None:
            s_label = key
            s = df[s_label]
        
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
        if expr is None:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5)
            cb.set_label(c_label)
        else:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5,
                              ticks=[(i+0.5)/(len(expr)+1) for i in range(len(expr)+1)])
            plt.clim(0, 1)
            cb.set_label(c_label)
            cb.ax.set_xticklabels(["Unassigned"] + expr)
            cb.ax.set_yticklabels(["Unassigned"] + expr)
    
    if show_legend:
        proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
        ax.legend([proxy],
                  [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
                  handler_map={mpatches.Circle: HandlerSizeLegend()})
    
    if interactive:  
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

def joint(model, data, x, y, **kwargs):
    df = to_dataframe(model, data)
    
    sns.jointplot(df[x],
                  df[y],
                  **kwargs)

def pairs(model, data,
          expr = None,
          class_label = "class",
          **kwargs):
    df = to_dataframe(model, data)
    
    if expr is None:
        sns.pairplot(df, **kwargs)
    else:
        df[class_label] = ["unassigned"]*df.shape[0]
        
        if isinstance(expr, six.string_types):
            expr = [expr]
            
        for e in expr:
            bin = df.query(e)
            df.loc[bin.index, class_label] = e
            
        sns.pairplot(df, hue=class_label, **kwargs)
     
    
    
def kdeplot(model, data, x, y,
            expr = None,
            alpha=1.0,
            cmap = ["Reds", "Blues", "Oranges", "Greens", "Greys"],
            **kwargs):
    df = to_dataframe(model, data)
    
    if expr is None:
        sns.kdeplot(df[x],
                    df[y],
                    cmap=cmap[0],
                    shade=True,
                    shade_lowest=False,
                    alpha=alpha,
                    **kwargs)
        
        proxy = mpatches.Circle((0.5, 0.5),
                                0.25,
                                fc=sns.color_palette(cmap[0])[-2])
        
        ax = plt.gca()
        ax.legend([proxy], ["Density"], **kwargs)
    else:
        proxies = []
        
        if isinstance(expr, six.string_types):
            expr = [expr]
            
        for i, e in reversed(list(enumerate(expr))):
            bin = df.query(e)
            sns.kdeplot(bin[x],
                        bin[y],
                        cmap=cmap[i % len(cmap)],
                        shade=True,
                        shade_lowest=False,
                        alpha=alpha,
                        **kwargs)
            proxies.append(mpatches.Circle((0.5, 0.5),
                                           0.25,
                                           fc=sns.color_palette(cmap[i % len(cmap)])[-2]))
            
        ax = plt.gca()
        ax.legend(proxies, expr, **kwargs)
        
def hist(model, data):
    df = to_dataframe(model, data)
    keys = model.responses.keys()
    
    f, axes = plt.subplots(1, len(keys))
    sns.despine(left=True)
    
    for i, k in enumerate(keys):
        sns.distplot(df[k], kde=False, ax=axes[i])
        
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    
def interact(model, data, x, y, z, **kwargs):
    df = to_dataframe(model, data)
    
    sns.interactplot(x, y, z, df, **kwargs)