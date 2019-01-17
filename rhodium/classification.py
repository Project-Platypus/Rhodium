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
import pydot
import operator
import functools
import itertools
import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO
from distutils.version import StrictVersion
from prim import Prim
from io import BytesIO

class Cart(object):
    
    def __init__(self,
                 x,
                 y, 
                 threshold = None, 
                 threshold_type = ">",
                 include = None,
                 exclude = None,
                 **kwargs):
        """Generates a decision tree for classification.
        
        Parameters
        ----------
        x : a matrix-like object (pandas.DataFrame, numpy.recarray, etc.)
            the independent variables
        y : a list-like object, the column name (str), or callable
            the dependent variable either provided as a list-like object
            classifying the data into cases of interest (e.g., False/True),
            a list-like object storing the raw variable value (in which case
            a threshold must be given), a string identifying the dependent
            variable in x, or a function called on each row of x to compute the
            dependent variable
        threshold : float
            threshold for identifying cases of interest
        threshold_type : str
            comparison operator used when identifying cases of interest
        include : list of str
            the names of variables included in the PRIM analysis
        exclude : list of str
            the names of variables excluded from the PRIM analysis
        """
        super(Cart, self).__init__()
        
        # Ensure the input x is a numpy matrix/array
        if isinstance(x, pd.DataFrame):
            x = x.to_records(index=False)
        elif isinstance(x, np.ma.MaskedArray):
            pass
        else:
            x = pd.DataFrame(x).to_records(index=False)
            
        # if y is a string or function, compute the actual response value
        # otherwise, ensure y is a numpy matrix/array
        if isinstance(y, six.string_types):
            key = y
            y = x[key]
            
            if exclude:
                exclude = list(exclude) + [key]
            else:
                exclude = [key]
        elif six.callable(y):
            fun = y
            y = np.apply_along_axis(fun, 0, x)
        elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, np.ma.MaskedArray):
            pass
        else:
            y = np.asarray(y)
            
        # convert include/exclude arguments to lists if they are strings
        if include and isinstance(include, six.string_types):
            include = [include]
            
        if exclude and isinstance(exclude, six.string_types):
            exclude = [exclude]     
            
        # include or exclude columns from the analysis
        if include:
            if isinstance(include, six.string_types):
                include = [include]

            drop_names = set(rf.get_names(x.dtype))-set(include)
            x = rf.drop_fields(x, drop_names, asrecarray=True)
        
        if exclude:
            if isinstance(exclude, six.string_types):
                exclude = [exclude]

            drop_names = set(exclude) 
            x = rf.drop_fields(x, drop_names, asrecarray=True)
            
        # apply the threshold if 
        if threshold:
            if six.callable(threshold):
                y = np.apply_along_axis(threshold, 0, y)
            else:
                # The syntax for threshold_type is "x <op> <threshold>", e.g.,
                # "x > 0.5".  However, partial only supports positional
                # arguments for built-in operators.  Thus, we must assign the
                # threshold to the first position and use a different operator.
                # For example, "x > 0.5" must be evaluated as "0.5 < x".
                OPERATORS = {"<" : operator.ge,
                             ">" : operator.le,
                             "<=" : operator.gt,
                             ">=" : operator.lt,
                             "=" : operator.eq}
                
                op = OPERATORS[threshold_type]
                y = np.apply_along_axis(functools.partial(op, threshold), 0, y)
                
        # validate inputs
        if len(y.shape) > 1:
            raise ValueError("y is not a 1-d array")
        
        # extract feature names
        feature_names = rf.get_names(x.dtype)
        
        # ensure x is formatted as a 2D matrix
        x = x.view("<f8").reshape(x.shape + (-1,))
        
        clf = tree.DecisionTreeClassifier(**kwargs)
        clf = clf.fit(x, y)
        
        # add our custom metadata to the classifier
        self._feature_names = feature_names
        self._x = x
        self._y = y
        self._clf = clf
    
    def _get_names(self, **kwargs):
        clf = self._clf
        feature_names = kwargs.get("feature_names", None)
        class_names = kwargs.get("class_names", None)
        
        if feature_names is None:
            feature_names = self._feature_names
            
        if class_names is None:
            class_names = clf.classes_
            
        return feature_names, class_names
        
    def _create_graph(self, **kwargs):
        clf = self._clf
        dot_data = StringIO()
        feature_names, class_names = self._get_names(**kwargs)
    
        if StrictVersion(sklearn.__version__) >= StrictVersion('0.17'):
            tree.export_graphviz(clf,
                                 out_file=dot_data,  
                                 feature_names=feature_names,
                                 class_names=class_names,  
                                 filled=kwargs.get("filled", True),
                                 rounded=kwargs.get("rounded", True),  
                                 special_characters=kwargs.get("special_characters", True),
                                 **kwargs)
        else:
            tree.export_graphviz(clf,
                                 out_file=dot_data,  
                                 feature_names=feature_names,
                                 **kwargs)
    
        return pydot.graph_from_dot_data(dot_data.getvalue())[0]
    
    def __str__(self):
        return self._to_string()
        
    def print_tree(self, coi=None, all=True, **kwargs):
        print(self._to_string(coi, all, **kwargs))
    
    def _to_string(self, coi=None, all=True, **kwargs):
        result = ""
        clf = self._clf
        feature_names, class_names = self._get_names(**kwargs)
        
        if not hasattr(coi, "__iter__") and not isinstance(coi, six.string_types):
            coi = [coi]
        
        left      = clf.tree_.children_left
        right     = clf.tree_.children_right
        threshold = clf.tree_.threshold
        features  = [feature_names[i] for i in clf.tree_.feature]
        classes   = [class_names[np.argmax(i)] for i in clf.tree_.value]
        
        # get ids of the nodes to print
        if all:
            idx = range(1, clf.tree_.node_count)
        else:
            idx = np.argwhere(left == -1)[:,0]     
    
        def recurse(left, right, child, lineage=None):
            if lineage is None:
                lineage = []
            if child in left:
                parent = np.where(left == child)[0].item()
                split = "l"
            elif child in right:
                parent = np.where(right == child)[0].item()
                split = "r"
    
            lineage.append((features[parent], "<=" if split=="l" else ">", threshold[parent]))
    
            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
    
        for child in idx:
            if coi is None or classes[child] in coi:
                if len(result) > 0:
                    result += "\n"
                
                result += "Node %d: %s\n" % (child, classes[child])
                
                if coi is not None:
                    value = clf.tree_.value[child][0]
                    ncoi = sum([value[i] if class_names[i] in coi else 0 for i in range(clf.n_classes_)])
                    density = ncoi/np.sum(value)
                    coverage = ncoi/sum([1 if yi in coi else 0 for yi in self._y])
                    result += "    Density: %.2f%%\n" % (100*density,)
                    result += "    Coverage: %.2f%%\n" % (100*coverage,)
                
                result += "    Rule: " + " and\n          ".join(self._collapse_bounds(recurse(left, right, child), feature_names))
                
        return result
                
    def _collapse_bounds(self, rules, keys):
        bounds = {}
        
        for rule in rules:
            if rule[0] in bounds:
                if rule[1] == "<=" and rule[2] <= bounds[rule[0]][1]:
                    bounds[rule[0]][1] = rule[2]
                elif rule[1] == ">" and rule[2] > bounds[rule[0]][0]:
                    bounds[rule[0]][0] = rule[2]
            else:
                bounds[rule[0]] = [rule[2] if rule[1] == ">" else -np.inf,
                                   rule[2] if rule[1] == "<=" else np.inf]
                
        result = []
        
        for key in keys:
            if key in bounds:
                if np.isinf(bounds[key][0]):
                    rule = "%s <= %f" % (key, bounds[key][1])
                elif np.isinf(bounds[key][1]):
                    rule = "%s > %f" % (key, bounds[key][0])
                else:
                    rule = "%f <= %s <= %f" % (bounds[key][0], key, bounds[key][1])
    
                result.append(rule)
                
        return result
        
    def show_tree(self, **kwargs):
        graph = self._create_graph(**kwargs)
        
        if "inline" in mpl.get_backend():
            # running inline in IPython
            from IPython.display import Image  
            return Image(graph.create_png())
        else:
            # otherwise show within matplotlib
            img_data = BytesIO(graph.create_png())
            img = mpimg.imread(img_data)
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            return fig
        
    def save(self, file, format="png", **kwargs):
        graph = self._create_graph(**kwargs) 
        graph.write(file, format=format)
        
    def save_pdf(self, file, feature_names=None, **kwargs):
        self.save(file, "pdf", feature_names, **kwargs)
        
    def save_png(self, file, feature_names=None, **kwargs):
        self.save(file, "png", feature_names, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._clf, name)
    