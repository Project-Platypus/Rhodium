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
import inspect
import operator
import itertools
from SALib.sample import saltelli, fast_sampler, finite_diff, latin
from SALib.analyze import sobol, fast, dgsm, delta
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
from SALib.sample import ff as ff_sampler
from SALib.analyze import ff as ff_analyzer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .model import *
from .optimization import *
from .sampling import *

def _cleanup_kwargs(function, kwargs):
    argspec = inspect.getargspec(function)
    result = {}
    
    if not argspec.keywords:
        for key in kwargs.keys():
            if key in argspec.args or hasattr(argspec, "kwargs"):
                result[key] = kwargs[key]
                
    return result

def _S2_to_dict(matrix, problem):
    result = {}
    names = list(problem["names"])
    
    for i in range(problem["num_vars"]):
        for j in range(i+1, problem["num_vars"]):
            if names[i] not in result:
                result[names[i]] = {}
            if names[j] not in result:
                result[names[j]] = {}
                
            result[names[i]][names[j]] = result[names[j]][names[i]] = float(matrix[i][j])
            
    return result

def _predict_N(method, nsamples, nvars, kwargs):
    if method == "sobol":
        if kwargs.get("calc_second_order", True):
            return int(math.ceil(nsamples / (2*nvars+2)))
        else:
            return int(math.ceil(nsamples / (nvars+2)))
    elif method == "morris":
        return int(math.ceil(nsamples / (nvars+1)))
    elif method == "fast":
        return int(math.ceil(nsamples / nvars))
    elif method == "dgsm":
        return int(math.ceil(nsamples / (nvars+1)))
    elif method == "delta":
        return nsamples
    else:
        return nsamples
    
class SAResult(dict):
    
    def __init__(self, parameters, *args, **kwargs):
        super(SAResult, self).__init__(*args, **kwargs)
        self.parameters = parameters
        
    def _longest_name(self):
        return max(len(k) for k in self.parameters)
                
    def __str__(self):
        lines = []
        format_str = "    %" + str(self._longest_name()) + "s: %+8f"
        format_conf = " (%+8f)"
        format_second = "    %" + str(2*self._longest_name()+3) + "s: %+8f"

        if "S1" in self:
            lines.append("First order sensitivity indices (confidence interval):")
            for k in self.parameters:
                line = format_str % (k, self["S1"][k])
                if "S1_conf" in self:
                    line += format_conf % self["S1_conf"][k]
                lines.append(line)
                
        if "ST" in self:
            lines.append("Total order sensitivity indices (confidence interval):")
            for k in self.parameters:
                line = format_str % (k, self["ST"][k])
                if "S1_conf" in self:
                    line += format_conf % self["ST_conf"][k]
                lines.append(line)
                
        if "S2" in self:
            lines.append("Second order sensitivity indices (confidence interval):")
            for k1, k2 in itertools.combinations(self.parameters, 2):
                line = format_second % (k1 + " - " + k2, self["S2"][k1][k2])
                if "S2_conf" in self:
                    line += format_conf % self["S2_conf"][k1][k2]
                lines.append(line)
                
        if "delta" in self:
            lines.append("Borgonovo's delta moment (confidence interval):")
            for k in self.parameters:
                line = format_str % (k, self["delta"][k])
                if "delta_conf" in self:
                    line += format_conf % self["delta_conf"][k]
                lines.append(line)
                
        if "vi" in self:
            lines.append("DGSM's Importance Criteria (stdev):")
            for k in self.parameters:
                line = format_str % (k, self["vi"][k])
                if "vi_std" in self:
                    line += format_conf % self["vi_std"][k]
                lines.append(line)
                
        if "dgsm" in self:
            lines.append("DGSM's Sensitivity Index (confidence interval):")
            for k in self.parameters:
                line = format_str % (k, self["dgsm"][k])
                if "dgsm_conf" in self:
                    line += format_conf % self["dgsm_conf"][k]
                lines.append(line)
                
        if "mu" in self:
            lines.append("Morris Method's mu:")
            for k in self.parameters:
                lines.append(format_str % (k, self["mu"][k]))
                
        if "mu_star" in self:
            lines.append("Morris Method's mu* (confidence interval):")
            for k in self.parameters:
                line = format_str % (k, self["mu_star"][k])
                if "mu_star_conf" in self:
                    line += format_conf % self["mu_star_conf"][k]
                lines.append(line)
                
        if "sigma" in self:
            lines.append("Morris Method's sigma:")
            for k in self.parameters:
                lines.append(format_str % (k, self["sigma"][k]))
        
        return "\n".join(lines)
    
    def plot(self, **kwargs):
        nfigs = sum([k in ("S1", "ST", "delta") for k in self.keys()])
        fig, axarr = plt.subplots(1, nfigs, sharey=True)
        n = len(self.parameters)
        index = 0
        
        if "S1" in self:
            axarr[index].bar(range(n),
                             [self["S1"][k] for k in self.parameters],
                             yerr=[self["S1_conf"][k] for k in self.parameters] if "S1_conf" in self else None,
                             align="center")
            axarr[index].set_title("First order sensitivity indices")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        if "ST" in self:
            axarr[index].bar(range(n),
                             [self["ST"][k] for k in self.parameters],
                             yerr=[self["ST_conf"][k] for k in self.parameters] if "ST_conf" in self else None,
                             align="center")
            axarr[index].set_title("Total order sensitivity indices")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        if "delta" in self:
            axarr[index].bar(range(n),
                             [self["delta"][k] for k in self.parameters],
                             yerr=[self["delta_conf"][k] for k in self.parameters] if "delta_conf" in self else None,
                             align="center")
            axarr[index].set_title("Borgonovo's delta moment")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        if "vi" in self:
            axarr[index].bar(range(n),
                             [self["vi"][k] for k in self.parameters],
                             yerr=[self["vi_std"][k] for k in self.parameters] if "vi_std" in self else None,
                             align="center")
            axarr[index].set_title("DGSM's Importance Criteria")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        if "dgsm" in self:
            axarr[index].bar(range(n),
                             [self["dgsm"][k] for k in self.parameters],
                             yerr=[self["dgsm_conf"][k] for k in self.parameters] if "dgsm_conf" in self else None,
                             align="center")
            axarr[index].set_title("DGSM's Sensitivity Index")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        if "mu" in self:
            axarr[index].bar(range(n),
                             [self["mu"][k] for k in self.parameters],
                             align="center")
            axarr[index].set_title("Morris Method's mu")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1

        if "mu_star" in self:
            axarr[index].bar(range(n),
                             [self["mu_star"][k] for k in self.parameters],
                             yerr=[self["mu_star_conf"][k] for k in self.parameters] if "mu_star_conf" in self else None,
                             align="center")
            axarr[index].set_title("Morris Method's mu*")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1

        if "sigma" in self:
            axarr[index].bar(range(n),
                             [self["sigma"][k] for k in self.parameters],
                             align="center")
            axarr[index].set_title("Morris Method's sigma")
            axarr[index].set_xticks(range(n))
            axarr[index].set_xticklabels(self.parameters)
            index += 1
            
        return fig
    
    def _is_significant(self, value, confidence_interval, threshold="conf"):
        if threshold == "conf":
            return value - abs(confidence_interval) > 0
        else:
            return value - abs(float(threshold)) > 0
    
    def plot_sobol(self, radSc=2.0, scaling=1, widthSc=0.5, STthick=1, varNameMult=1.3, colors=None, groups=None, gpNameMult=1.5, threshold="conf"):
        # Derived from https://github.com/calvinwhealton/SensitivityAnalysisPlots
        fig, ax = plt.subplots(1, 1)
        color_map = {}
        
        # initialize parameters and colors
        if groups is None:
            parameters = list(self.parameters)
            
            if colors is None:
                colors = ["k"]
            
            for i, parameter in enumerate(parameters):
                color_map[parameter] = colors[i % len(colors)]
        else:
            parameters = []
            
            if colors is None:
                colors = sns.color_palette("deep", max(3, len(groups)))
            
            for i, key in enumerate(groups.keys()):
                parameters.extend(groups[key])
                
                for parameter in groups[key]:
                    color_map[parameter] = colors[i % len(colors)]
        
        n = len(parameters)
        angles = radSc*math.pi*np.arange(0, n)/n
        x = radSc*np.cos(angles)
        y = radSc*np.sin(angles)
        
        # plot second-order indices
        for i, j in itertools.combinations(range(n), 2):
            key1 = parameters[i]
            key2 = parameters[j]
            
            if self._is_significant(self["S2"][key1][key2], self["S2_conf"][key1][key2], threshold):
                angle = math.atan((y[j]-y[i])/(x[j]-x[i]))
                    
                if y[j]-y[i] < 0:
                    angle += math.pi
                    
                line_hw = scaling*(max(0, self["S2"][key1][key2])**widthSc)/2
                    
                coords = np.empty((4, 2))
                coords[0, 0] = x[i] - line_hw*math.sin(angle)
                coords[1, 0] = x[i] + line_hw*math.sin(angle)
                coords[2, 0] = x[j] + line_hw*math.sin(angle)
                coords[3, 0] = x[j] - line_hw*math.sin(angle)
                coords[0, 1] = y[i] + line_hw*math.cos(angle)
                coords[1, 1] = y[i] - line_hw*math.cos(angle)
                coords[2, 1] = y[j] - line_hw*math.cos(angle)
                coords[3, 1] = y[j] + line_hw*math.cos(angle)
    
                ax.add_artist(plt.Polygon(coords, color="0.75"))
            
        # plot total order indices
        for i, key in enumerate(parameters):
            if self._is_significant(self["ST"][key], self["ST_conf"][key], threshold):
                ax.add_artist(plt.Circle((x[i], y[i]), scaling*(self["ST"][key]**widthSc)/2, color='w'))
                ax.add_artist(plt.Circle((x[i], y[i]), scaling*(self["ST"][key]**widthSc)/2, lw=STthick, color='0.4', fill=False))
        
        # plot first-order indices
        for i, key in enumerate(parameters):
            if self._is_significant(self["S1"][key], self["S1_conf"][key], threshold):
                ax.add_artist(plt.Circle((x[i], y[i]), scaling*(self["S1"][key]**widthSc)/2, color='0.4'))
               
        # add labels
        for i, key in enumerate(parameters):                
            ax.text(varNameMult*x[i], varNameMult*y[i], key, ha='center', va='center',
                    rotation=angles[i]*360/(2*math.pi) - 90,
                    color=color_map[key])
            
        if groups is not None:
            for i, group in enumerate(groups.keys()):
                group_angle = np.mean([angles[j] for j in range(n) if parameters[j] in groups[group]])
                
                ax.text(gpNameMult*radSc*math.cos(group_angle), gpNameMult*radSc*math.sin(group_angle), group, ha='center', va='center',
                    rotation=group_angle*360/(2*math.pi) - 90,
                    color=colors[i % len(colors)])
                
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('equal')
        plt.axis([-2*radSc, 2*radSc, -2*radSc, 2*radSc])
        
        return fig

def sa(model, response, policy={}, method="sobol", nsamples=1000, **kwargs):
    if len(model.uncertainties) == 0:
        raise ValueError("no uncertainties defined in model")
    
    problem = { 'num_vars' : len(model.uncertainties),
                'names' : model.uncertainties.keys(),
                'bounds' : [[0.0, 1.0] for u in model.uncertainties],
                'groups' : kwargs.get("groups", None) }

    # estimate the argument N passed to the sampler that produces the requested
    # number of samples
    N = _predict_N(method, nsamples, problem["num_vars"], kwargs)
    
    # generate the samples
    if method == "sobol":
        samples = saltelli.sample(problem, N, **_cleanup_kwargs(saltelli.sample, kwargs))
    elif method == "morris":
        samples = morris_sampler.sample(problem, N, **_cleanup_kwargs(morris_sampler.sample, kwargs))
    elif method == "fast":
        samples = fast_sampler.sample(problem, N, **_cleanup_kwargs(fast_sampler.sample, kwargs))
    elif method == "ff":
        samples = ff_sampler.sample(problem, **_cleanup_kwargs(ff_sampler.sample, kwargs))
    elif method == "dgsm":
        samples = finite_diff.sample(problem, N, **_cleanup_kwargs(finite_diff.sample, kwargs))
    elif method == "delta":
        if "samples" in kwargs:
            samples = kwargs["samples"]
        else:
            samples = latin.sample(problem, N, **_cleanup_kwargs(latin.sample, kwargs))
            
    # convert from samples in [0, 1] to uncertainty domain
    for i, u in enumerate(model.uncertainties):
        samples[:,i] = u.ppf(samples[:,i])
        
    # run the model and collect the responses
    responses = np.empty(samples.shape[0])
    
    for i in range(samples.shape[0]):
        sample = {k : v for k, v in zip(model.uncertainties.keys(), samples[i])}
        responses[i] = evaluate(model, overwrite(sample, policy))[response]
    
    # run the sensitivity analysis method
    if method == "sobol":
        result = sobol.analyze(problem, responses, **_cleanup_kwargs(sobol.analyze, kwargs))
    elif method == "morris":
        result = morris_analyzer.analyze(problem, samples, responses, **_cleanup_kwargs(morris_analyzer.analyze, kwargs))
    elif method == "fast":
        result = fast.analyze(problem, responses, **_cleanup_kwargs(fast.analyze, kwargs))
    elif method == "ff":
        result = ff_analyzer.analyze(problem, samples, responses, **_cleanup_kwargs(ff_analyzer.analyze, kwargs))
    elif method == "dgsm":
        result = dgsm.analyze(problem, samples, responses, **_cleanup_kwargs(dgsm.analyze, kwargs))
    elif method == "delta":
        result = delta.analyze(problem, samples, responses, **_cleanup_kwargs(delta.analyze, kwargs))
         
    # convert the SALib results into a form allowing pretty printing and
    # lookups using the parameter name
    pretty_result = SAResult(list(result["names"] if "names" in result else problem["names"]))
    
    if "S1" in result:
        pretty_result["S1"] = {k : float(v) for k, v in zip(problem["names"], result["S1"])}
    if "S1_conf" in result:
        pretty_result["S1_conf"] = {k : float(v) for k, v in zip(problem["names"], result["S1_conf"])}
    if "ST" in result:
        pretty_result["ST"] = {k : float(v) for k, v in zip(problem["names"], result["ST"])}
    if "ST_conf" in result:
        pretty_result["ST_conf"] = {k : float(v) for k, v in zip(problem["names"], result["ST_conf"])}
    if "S2" in result:
        pretty_result["S2"] = _S2_to_dict(result["S2"], problem)
    if "S2_conf" in result:
        pretty_result["S2_conf"] = _S2_to_dict(result["S2_conf"], problem)
    if "delta" in result:
        pretty_result["delta"] = {k : float(v) for k, v in zip(problem["names"], result["delta"])}
    if "delta_conf" in result:
        pretty_result["delta_conf"] = {k : float(v) for k, v in zip(problem["names"], result["delta_conf"])}
    if "vi" in result:
        pretty_result["vi"] = {k : float(v) for k, v in zip(problem["names"], result["vi"])}
    if "vi_std" in result:
        pretty_result["vi_std"] = {k : float(v) for k, v in zip(problem["names"], result["vi_std"])}
    if "dgsm" in result:
        pretty_result["dgsm"] = {k : float(v) for k, v in zip(problem["names"], result["dgsm"])}
    if "dgsm_conf" in result:
        pretty_result["dgsm_conf"] = {k : float(v) for k, v in zip(problem["names"], result["dgsm_conf"])}
    if "mu" in result:
        pretty_result["mu"] = {k : float(v) for k, v in zip(result["names"], result["mu"])}
    if "mu_star" in result:
        pretty_result["mu_star"] = {k : float(v) for k, v in zip(result["names"], result["mu_star"])}
    if "mu_star_conf" in result:
        pretty_result["mu_star_conf"] = {k : float(v) for k, v in zip(result["names"], result["mu_star_conf"])}
    if "sigma" in result:
        pretty_result["sigma"] = {k : float(v) for k, v in zip(result["names"], result["sigma"])}

    return pretty_result

def oat(model, response, policy={}, nsamples=100, **kwargs):
    keys = model.uncertainties.keys()
    responses = np.empty((nsamples, len(keys)))
    samples = np.linspace(0.1, 0.99, num=nsamples)
    base_response = evaluate(model, policy)[response]
    
    for i, u in enumerate(model.uncertainties):
        ppf_samples = u.ppf(samples)
        
        for j in range(nsamples):
            sample = { u.name : ppf_samples[j] }
            responses[j, i] = evaluate(model, overwrite(sample, policy))[response]

    minv = np.nanmin(responses, axis=0)
    maxv = np.nanmax(responses, axis=0)
    midv = responses[nsamples//2]
    
    total_range = maxv-minv
    negative_percentage = abs(midv - minv) / total_range
    positive_percentage = abs(maxv - midv) / total_range

    total_range /= np.max(total_range)
    negative_percentage *= total_range / 2
    positive_percentage *= total_range / 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # draw the lines first to get colors
    colors = []
    
    for i, u in enumerate(model.uncertainties):
        h = ax2.plot(samples, responses[:,i])
        colors.append(h[0].get_color())
        
    ax2.axvline(0.5, ls='--', color='k')
    ax2.axhline(base_response, ls='--', color='k')
    ax2.set_xlim([0.1, 0.99])
    ax2.set_xticks([0.1, 0.25, 0.5, 0.75, 0.99])
    ax2.set_xticklabels(["1", "25", "50", "75", "99"])
    ax2.set_xlabel("Quantile of Prior (%)")
    ax2.set_ylabel(response)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.legend(keys)
    
    # then draw the tornado chart  
    ax1.barh(range(len(keys)), negative_percentage, left=0.5-negative_percentage, align='center', color=colors)
    ax1.barh(range(len(keys)), positive_percentage, left=0.5, align='center', color=colors)
    ax1.set_yticks(range(len(keys)))
    ax1.set_yticklabels(keys)
    ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["-100%", "-50%", "0%", "50%", "100%"])
    ax1.set_xlabel("Percent of Total Variance")
    ax1.set_xlim([0, 1])
    
    return fig

def regional_sa(model, expr, policy={}, nsamples=1000):
    samples = sample_lhs(model, nsamples)
    output = evaluate(model, overwrite(samples, policy))
    classification = output.apply(expr)
    classes = sorted(set(classification))
    fig, axarr = plt.subplots(1, len(model.uncertainties))
    lines = []
    
    for i, u in enumerate(model.uncertainties):
        for k in classes:
            indices = [classification[j] == k for j in range(len(classification))]
            values = [output[j][u.name] for j in range(len(indices)) if indices[j]]
            sorted_values = sorted(enumerate(values), key=operator.itemgetter(1))
            h = axarr[i].plot([v[1] for v in sorted_values], np.arange(len(values))/float(len(values)-1))
            lines.append(h[0])
            
        values = [output[j][u.name] for j in range(len(indices))]
        sorted_values = sorted(enumerate(values), key=operator.itemgetter(1))
        h = axarr[i].plot([v[1] for v in sorted_values], np.arange(len(values))/float(len(values)-1))
        lines.append(h[0])
        
        axarr[i].set_title(u.name)
        
    plt.figlegend(lines[:len(classes)] + [lines[-1]],
                  list(map(str, classes)) + ["Unconditioned"],
                  loc='lower center',
                  ncol=3,
                  labelspacing=0. )
    
    return fig