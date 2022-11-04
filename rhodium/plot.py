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
import mplcursors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.colors import ColorConverter, Normalize
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.mplot3d import Axes3D
from .config import RhodiumConfig
from .model import Response
from .brush import Brush, BrushSet, apply_brush, color_brush, brush_color_map, color_indices

try:
    set
except NameError:
    from sets import Set as set  # @UnresolvedImport

def _combine_keys(*args):
    result = []
    result_set = set()
    
    for arg in args:
        if arg is None:
            continue
        
        if hasattr(arg, "__iter__") and not isinstance(arg, str):
            for key in arg:
                if key not in result_set:
                    result.append(key)
                    result_set.add(key)
        elif arg not in result_set:
            result.append(arg)
            result_set.add(arg)
            
    return result

class HandlerSizeLegend(HandlerPatch):
    def __call__(self, legend, orig_handle,
             fontsize,
             handlebox):
        print("here")
    
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
           brush = None,
           pick_handler = None,
           **kwargs):
    
    df = data.as_dataframe()
    
    if "axes.facecolor" in mpl.rcParams:
        orig_facecolor = mpl.rcParams["axes.facecolor"]
        mpl.rcParams["axes.facecolor"] = "white"
        
    if brush is not None:
        brush_set = BrushSet(brush)
        c, color_map = color_brush(brush_set, df)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(x, str):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, str):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(z, str):
        z_label = z
        z = df[z_label]
    else:
        z_label = None
        
    if isinstance(c, str):
        c_label = c
        c = df[c_label]
    else:
        c_label = None
        
    if isinstance(s, str):
        s_label = s
        s = df[s_label]
    else:
        s_label = None
        
    used_keys = set([x_label, y_label, z_label, c_label, s_label, None])
    used_keys.remove(None)
    
    remaining_keys = list(model.responses.keys())

    for key in used_keys:
        if key in remaining_keys:
            remaining_keys.remove(key)

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

    if "cmap" not in kwargs:
        kwargs["cmap"] = RhodiumConfig.default_cmap

    handle = ax.scatter(xs = x,
                        ys = y,
                        zs = z,
                        c = c,
                        s = s,
                        picker = kwargs["picker"] if "picker" in kwargs else pick_handler is not None,
                        **kwargs)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
        
    if show_colorbar:
        if brush is None:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5)
            cb.set_label(c_label)
        else:
            handle.set_array(np.asarray(color_indices(c, color_map)))
            handle.cmap = mpl.colors.ListedColormap(list(color_map.values()))
            off = (len(color_map)-1)/(len(color_map))/2
            height = (len(color_map)-1)-2*off
            ticks = [0] if len(color_map) <= 1 else [(i/(len(color_map)-1) * height + off) for i in range(len(color_map))]
            cb = fig.colorbar(handle, shrink=0.5, aspect=5, ticks=ticks)
            cb.set_label("")
            cb.ax.set_xticklabels(color_map.keys())
            cb.ax.set_yticklabels(color_map.keys())
    
    if show_legend:
        proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
        ax.legend([proxy],
                  [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
                  handler_map={proxy: HandlerSizeLegend()})
        
    if interactive:
        def formatter(sel):
            i = sel.index
            point = data[i]
            keys = model.responses.keys()
            label = "Index %d" % i
            
            for key in keys:
                label += "\n%s: %0.2f" % (key, point[key])
            
            return label
            
        c = mplcursors.cursor(handle, hover=True)
        c.connect(
            "add", lambda sel: sel.annotation.set_text(formatter(sel)))
        
    if pick_handler:
        def handle_click(event):
            if hasattr(event, "ind"):
                i = event.ind[0]
                pick_handler(i)
                plt.draw()
        
        fig.canvas.mpl_connect('pick_event', handle_click)
    
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
           brush = None,
           is_class = False,
           colors = None,
           **kwargs):
    df = data.as_dataframe()
    fig = plt.figure(facecolor='white')
    ax = plt.gca()
    
    if brush is not None:
        brush_set = BrushSet(brush)
        c, color_map = color_brush(brush_set, df)
    
    if isinstance(x, str):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, str):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(c, str):
        c_label = c
        c = df[c_label]
    else:
        c_label = None
        
    if isinstance(s, str):
        s_label = s
        s = df[s_label]
    else:
        s_label = None
        
    used_keys = set([x_label, y_label, c_label, s_label, None])
    used_keys.remove(None)
    
    remaining_keys = list(model.responses.keys())

    for key in used_keys:
        if key in remaining_keys:
            remaining_keys.remove(key)

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
        

    if is_class:
        if isinstance(colors, dict):
            cmap = colors
        else:
            from pandas.tools.plotting import _get_standard_colors
            classes = c.drop_duplicates()
            color_values = _get_standard_colors(num_colors=len(classes),
                                                colormap=kwargs["cmap"] if "cmap" in kwargs else None,
                                                color_type='random',
                                                color=colors)
            cmap = dict(zip(classes, color_values))
        c = [cmap[c_i] for c_i in c]
        show_colorbar = False
    elif "cmap" not in kwargs:
        kwargs["cmap"] = RhodiumConfig.default_cmap  
    

    handle = plt.scatter(x = x,
                         y = y,
                         c = c,
                         s = s,
                         **kwargs)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
        
    if show_colorbar:
        if brush is None:
            cb = fig.colorbar(handle, shrink=0.5, aspect=5)
            cb.set_label(c_label)
        else:
            handle.set_array(np.asarray(color_indices(c, color_map)))
            handle.cmap = mpl.colors.ListedColormap(list(color_map.values()))
            off = (len(color_map)-1)/(len(color_map))/2
            height = (len(color_map)-1)-2*off
            ticks = [0] if len(color_map) <= 1 else [(i/(len(color_map)-1) * height + off) for i in range(len(color_map))]
            cb = fig.colorbar(handle, shrink=0.5, aspect=5, ticks=ticks)
            cb.set_label("")
            cb.ax.set_xticks(ticks)
            cb.ax.set_xticklabels(color_map.keys())
            cb.ax.set_yticklabels(color_map.keys())
    
    if show_legend:
        proxy = mpatches.Circle((0.5, 0.5), 0.25, fc="b")
        ax.legend([proxy],
                  [s_label + " (" + str(s_min) + " - " + str(s_max) + ")"],
                  handler_map={mpatches.Circle: HandlerSizeLegend()})
    
    if interactive:  
        def formatter(sel):
            i = sel.index
            point = data[i]
            keys = model.responses.keys()
            label = "Index %d" % i
            
            for key in keys:
                label += "\n%s: %0.2f" % (key, point[key])
            
            return label
            
        c = mplcursors.cursor(handle, hover=True)
        c.connect(
            "add", lambda sel: sel.annotation.set_text(formatter(sel)))
        
    return fig

def joint(model, data, x, y, **kwargs):
    df = data.as_dataframe(_combine_keys(x, y))
    
    sns.jointplot(x=df[x],
                  y=df[y],
                  **kwargs)

def pairs(model, data,
          keys = None,
          brush = None,
          brush_label = "class",
          **kwargs):
    df = data.as_dataframe(keys if keys is not None else model.responses.keys())
    
    if brush is None:
        sns.pairplot(df, **kwargs)
    else:
        brush_set = BrushSet(brush)
        df[brush_label] = apply_brush(brush_set, df)
        
        if "palette" not in kwargs:
            kwargs["palette"] = brush_color_map(brush_set, df[brush_label])
            
        sns.pairplot(df, hue=brush_label, **kwargs)
     
def kdeplot(model, data, x, y,
            brush = None,
            alpha=1.0,
            cmap = ["Reds", "Blues", "Oranges", "Greens", "Greys"],
            **kwargs):
    df = data.as_dataframe()
    
    if brush is None:
        sns.kdeplot(x=df[x],
                    y=df[y],
                    cmap=cmap[0],
                    fill=True,
                    thresh=0.05,
                    alpha=alpha,
                    **kwargs)
        
        proxy = mpatches.Circle((0.5, 0.5),
                                0.25,
                                fc=sns.color_palette(cmap[0])[-2])
        
        ax = plt.gca()
        ax.legend([proxy], ["Density"], **kwargs)
    else:
        proxies = []
        brush_set = BrushSet(brush)
            
        for i, brush in reversed(list(enumerate(brush_set))):
            bin = df.query(brush.expr)
            sns.kdeplot(x=bin[x],
                        y=bin[y],
                        cmap=cmap[i % len(cmap)],
                        fill=True,
                        thresh=0.05,
                        alpha=alpha,
                        **kwargs)
            proxies.append(mpatches.Circle((0.5, 0.5),
                                           0.25,
                                           fc=sns.color_palette(cmap[i % len(cmap)])[-2]))
            
        ax = plt.gca()
        ax.legend(proxies, brush_set.keys(), **kwargs)
        
def hist(model, data, keys=None):
    df = data.as_dataframe(model.responses.keys() if keys is None else _combine_keys(keys))
    keys = model.responses.keys()
    
    f, axes = plt.subplots(1, len(keys))
    sns.despine(left=True)
    
    for i, k in enumerate(keys):
        sns.distplot(df[k], kde=False, ax=axes[i])
        
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    
def interact(model, data, x, y, z, **kwargs):
    df = data.as_dataframe(_combine_keys(x, y, z))
    sns.interactplot(x, y, z, df, **kwargs)
    
def contour2d(model, data, x=None, y=None, z=None, levels=15, size=100, xlim=None, ylim=None, labels=True, show_colorbar=True, shrink=0.05, method='cubic', **kwargs):
    df = data.as_dataframe(_combine_keys(model.responses.keys(), x, y, z))
    
    if isinstance(x, str):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, str):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(z, str):
        z_label = z
        z = df[z_label]
    else:
        z_label = None
        
    used_keys = set([x_label, y_label, z_label, None])
    used_keys.remove(None)
    
    remaining_keys = list(model.responses.keys())

    for key in used_keys:
        if key in remaining_keys:
            remaining_keys.remove(key)

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

    # compute the grid
    if xlim is None:
        xmin = np.min(x)
        xmax = np.max(x)
    else:
        xmin = xlim[0]
        xmax = xlim[1]
        
    if ylim is None:
        ymin = np.min(y)
        ymax = np.max(y)
    else:
        ymin = ylim[0]
        ymax = ylim[1]

    # grid the data
    xshrink = shrink*(xmax-xmin)
    yshrink = shrink*(ymax-ymin)
    xi, yi = np.mgrid[(xmin+xshrink):(xmax-xshrink):complex(size), (ymin+yshrink):(ymax-yshrink):complex(size)]
    x = x.values
    y = y.values
    z = z.values
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    points = np.concatenate((x, y), axis=1)
    
    zi = griddata(points, z, (xi, yi), method=method)
    
    # generate the plot
    fig = plt.figure(facecolor='white')
    ax = plt.gca()
    
    if labels:
        CS = plt.contour(xi, yi, zi, levels, colors="k", linewidths=0.5)
        plt.clabel(CS, **kwargs)
    
    if "cmap" not in kwargs:
        kwargs["cmap"] = RhodiumConfig.default_cmap
    
    plt.contourf(xi, yi, zi, levels, **kwargs)
    ax.set_xlim(xmin+xshrink, xmax-xshrink)
    ax.set_ylim(ymin+yshrink, ymax-yshrink)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # draw colorbar
    if show_colorbar:
        cb = plt.colorbar()
        cb.set_label(z_label)

    return fig

def contour3d(model, data, x=None, y=None, z=None, xlim=None, ylim=None, levels=15, size=100, show_colorbar=True, shrink=0.05, method='cubic', **kwargs):
    df = data.as_dataframe(_combine_keys(model.responses.keys(), x, y, z))
    
    if len(df.columns) < 3:
        raise ValueError("insufficient number of dimensions")
    
    if "axes.facecolor" in mpl.rcParams:
        orig_facecolor = mpl.rcParams["axes.facecolor"]
        mpl.rcParams["axes.facecolor"] = "white"
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(x, str):
        x_label = x
        x = df[x_label]
    else:
        x_label = None
            
    if isinstance(y, str):
        y_label = y
        y = df[y_label]
    else:
        y_label = None
        
    if isinstance(z, str):
        z_label = z
        z = df[z_label]
    else:
        z_label = None
        
    used_keys = set([x_label, y_label, z_label, None])
    used_keys.remove(None)
    
    remaining_keys = list(model.responses.keys())

    for key in used_keys:
        if key in remaining_keys:
            remaining_keys.remove(key)

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
            
    # compute the grid
    if xlim is None:
        xmin = np.min(x)
        xmax = np.max(x)
    else:
        xmin = xlim[0]
        xmax = xlim[1]
        
    if ylim is None:
        ymin = np.min(y)
        ymax = np.max(y)
    else:
        ymin = ylim[0]
        ymax = ylim[1]

    # grid the data
    xshrink = shrink*(xmax-xmin)
    yshrink = shrink*(ymax-ymin)
    xi, yi = np.mgrid[(xmin+xshrink):(xmax-xshrink):complex(size), (ymin+yshrink):(ymax-yshrink):complex(size)]
    x = x.values
    y = y.values
    z = z.values
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    points = np.concatenate((x, y), axis=1)
    
    zi = griddata(points, z, (xi, yi), method=method)
    
    if "cmap" not in kwargs:
        kwargs["cmap"] = RhodiumConfig.default_cmap
    
    zmin = np.nanmin(zi)
    zmax = np.nanmax(zi)

    handle = ax.plot_surface(xi, yi, zi, rstride=int(size/levels), cstride=int(size/levels), vmin=zmin, vmax=zmax, **kwargs)
    
    if show_colorbar:
        cb = fig.colorbar(handle, shrink=0.5, aspect=5)
        cb.set_label(z_label)
    
    ax.set_xlim(xmin+xshrink, xmax-xshrink)
    ax.set_ylim(ymin+yshrink, ymax-yshrink)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    return fig
    
def animate3d(prefix, dir="images/", steps=36, transform=(10, 0, 0), **kwargs):
    import os
    import math
    import inspect
    from PIL import Image
    from images2gif import writeGif
  
    base_dir = os.path.join(dir, prefix)
      
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    ax = plt.gca()
    digits = int(math.log10(steps-1))+1
    files = []
      
    for n in range(steps):
        if inspect.isfunction(transform):
            transform(ax)
        else:
            ax.azim += transform[0]
            ax.elev += transform[1]
            ax.dist += transform[2]
            
        filename = os.path.join(base_dir, 'img' + str(n).zfill(digits) + '.png')
        plt.savefig(filename, bbox_inches='tight')
        files.append(filename)
        
    images = [Image.open(file) for file in files]
    file_path_name = os.path.join(dir, prefix + '.gif')
    writeGif(file_path_name, images, **kwargs)
    
def parallel_coordinates(model, data, c=None, cols=None, ax=None, colors=None,
                     use_columns=False, xticks=None, colormap=None,
                     target="top", brush=None, zorder=None, **kwds):
    if "axes.facecolor" in mpl.rcParams:
        orig_facecolor = mpl.rcParams["axes.facecolor"]
        mpl.rcParams["axes.facecolor"] = "white"
    
    df = data.as_dataframe(_combine_keys(model.responses.keys(), cols, c)) #, exclude_dtypes=["object"])
        
    if brush is not None:
        brush_set = BrushSet(brush)
        assignment = apply_brush(brush_set, data)
        color_map = brush_color_map(brush_set, assignment)
        class_col = pd.DataFrame({"class" : assignment})["class"]
        is_class = True
    else:
        if c is None:
            c = df.columns.values[-1]
        
        class_col = df[c]
        is_class = df.dtypes[c].name == "object"
        color_map = None
    
        if is_class:
            df = df.drop(c, axis=1)
            
            if c in cols:
                cols.remove(c)
        else:
            class_min = class_col.min()
            class_max = class_col.max()
        
    if cols is not None:
        df = df[cols]
    
    df_min = df.min()
    df_max = df.max()
    
    df = (df - df_min) / (df_max - df_min)
    n = len(df)

    used_legends = set([])

    ncols = len(df.columns)
    
    for i in range(ncols):
        if target == "top":
            if model.responses[df.columns.values[i]].dir == Response.MINIMIZE:
                df.iloc[:,i] = 1-df.iloc[:,i]
        elif target == "bottom":
            if model.responses[df.columns.values[i]].dir == Response.MAXIMIZE:
                df.iloc[:,i] = 1-df.iloc[:,i]

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap(colormap)
    
    if is_class:
        if color_map is None:
            if isinstance(colors, dict):
                cmap = colors
            else:
                from pandas.tools.plotting import _get_standard_colors
                classes = class_col.drop_duplicates()
                color_values = _get_standard_colors(num_colors=len(classes),
                                                colormap=colormap, color_type='random',
                                                color=colors)
                cmap = dict(zip(classes, color_values))
        else:
            cmap = color_map
            
    if zorder is None:
        indices = range(n)
    else:
        indices = [i[0] for i in sorted(enumerate(df[zorder]), key=lambda x : x[1])]

    for i in indices:
        y = df.iloc[i].values
        kls = class_col.iat[i]
        
        if is_class:
            label = str(kls)
            
            if label not in used_legends:
                used_legends.add(label)
                ax.plot(x, y, label=label, color=cmap[kls], **kwds)
            else:
                ax.plot(x, y, color=cmap[kls], **kwds)
        else:
            ax.plot(x, y, color=cmap((kls - class_min)/(class_max-class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=2, color='black')
        format = "%.2f"
        
        if target == "top":
            value = df_min[i] if model.responses[df.columns.values[i]].dir == Response.MINIMIZE else df_max[i]
            
            if model.responses[df.columns.values[i]].dir != Response.INFO:
                format = format + "*"
        elif target == "bottom":
            value = df_max[i] if model.responses[df.columns.values[i]].dir == Response.MINIMIZE else df_min[i]
        else:
            value = df_max[i]
            
            if model.responses[df.columns.values[i]].dir == Response.MAXIMIZE:
                format = format + "*"
            
        ax.text(i, 1.001, format % value, ha="center", fontsize=10)
        format = "%.2f"
            
        if target == "top":
            value = df_max[i] if model.responses[df.columns.values[i]].dir == Response.MINIMIZE else df_min[i]
        elif target == "bottom":
            value = df_min[i] if model.responses[df.columns.values[i]].dir == Response.MINIMIZE else df_max[i]
            
            if model.responses[df.columns.values[i]].dir != Response.INFO:
                format = format + "*"
        else:
            value = df_min[i]
            
            if model.responses[df.columns.values[i]].dir == Response.MINIMIZE:
                format = format + "*"
            
        ax.text(i, -0.001, format % value, ha="center", va="top", fontsize=10)

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0]-0.1, x[-1]+0.1)
    ax.tick_params(direction="out", pad=10)
    
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2)
    if target == "top":
        ax.text(-0.05, 0.5, "Target", ha="center", va="center", rotation=90, bbox=bbox_props, transform=ax.transAxes)
    elif target == "bottom":
        ax.text(-0.05, 0.5, "Target", ha="center", va="center", rotation=-90, bbox=bbox_props, transform=ax.transAxes)

    if is_class:
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
        fig.subplots_adjust(right=0.8)
    else:
        cax,_ = mpl.colorbar.make_axes(ax)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, spacing='proportional', norm=mpl.colors.Normalize(vmin=class_min, vmax=class_max), format='%.2f')
        cb.set_label(c)
        cb.set_clim(class_min, class_max)
    
    mpl.rcParams["axes.facecolor"] = orig_facecolor
    
    return fig