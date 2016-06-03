import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *
from rhodium.config import RhodiumConfig
from platypus import MapEvaluator

RhodiumConfig.default_evaluator = MapEvaluator()

def lake_problem(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows)
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(pollution_limit)
    reliability = 0.0

    for _ in xrange(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in xrange(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    intertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (max_P, utility, intertia, reliability)

model = Model(lake_problem)

# define all parameters to the model that we will be studying
model.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

# define the model outputs
model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

# define any constraints (can reference any parameter or response by name)
model.constraints = [Constraint("reliability >= 0.95")]

# some parameters are levers that we control via our policy
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]

if os.path.exists("data.txt"):
    with open("data.txt", "r") as f:
        output = json.load(f)
    output = DataSet(output)
else:
    output = optimize(model, "NSGAII", 1000)
 
    with open("data.txt", "w") as f:
        json.dump(output, f)
        
output.save("test.csv")
        
(model, output) = load("test.csv", parameters="pollution_limit")
   
# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

# # Use Seaborn settings for pretty plots
# sns.set()
# 
# # Plot the points in 2D space
# scatter2d(model, output)
# plt.show()
# 
# # The optional interactive flag will show additional details of each point when
# # hovering the mouse
# scatter2d(model, output, brush="reliability >= 0.5 and utility > 0.5")
# plt.show()
# 
# # Most of Rhodiums's plotting functions accept an optional expr argument for
# # classifying or highlighting points meeting some condition
# scatter2d(model, output, x="reliability", brush=Brush("reliability >= 0.2"))
# plt.show()
#
# # Plot the points in 3D space
# scatter3d(model, output, s="reliability", show_legend=True)
# plt.show()
#     
# # Kernel density estimation plots show density contours for samples.  By
# # default, it will show the density of all sampled points
# kdeplot(model, output, x="max_P", y="utility")
# plt.show()
# 
# # Alternatively, we can show the density of all points meeting one or more
# # conditions
# kdeplot(model, output, x="max_P", y="utility",
#         brush=["reliability >= 0.2", "reliability < 0.2"],
#         alpha=0.8)
# plt.show()
# 
# # Pairwise scatter plots shown 2D scatter plots for all outputs
# pairs(model, output)
# plt.show()
# 
# # We can also highlight points meeting one or more conditions
# pairs(model, output,
#       brush=["reliability >= 0.2", "reliability < 0.2"])
# plt.show()
# 
# # Joint plots show a single pair of parameters in 2D, their distributions using
# # histograms, and the Pearson correlation coefficient
# joint(model, output, x="max_P", y="utility")
# sns.plt.show()
#
# # Interaction plots show the interaction between two parameters (x and y) with
# # respect to a response (z)
# interact(model, output, x="max_P", y="utility", z="reliability", filled=True)
# sns.plt.show()
#
# # A histogram of the distribution of points along each parameter
# hist(model, output)
# sns.plt.show()

# # A parallel coordinates plot to view interactions among responses
parallel_coordinates(model, output, colormap="rainbow", zorder="reliability", brush=Brush("reliability > 0.2"))     
plt.show()

# ----------------------------------------------------------------------------
# Identifying Key Uncertainties
# ----------------------------------------------------------------------------

# The remaining figures look better using Matplotlib's default settings
#mpl.rcdefaults()

# We can manually construct policies for analysis.  A policy is simply a Python
# dict storing key-value pairs, one for each lever.
#policy = {"pollution_limit" : [0.02]*100}

# Or select one of our optimization results
#policy = output[3]

# construct a specific policy and evaluate it against 1000 states-of-the-world
#SOWs = sample_lhs(model, 100)
#results = evaluate(model, update(SOWs, policy))
#metric = ["Reliable" if v["reliability"] > 0.9 else "Unreliable" for v in results]
 
# # use PRIM to identify the key uncertainties if we require reliability > 0.9
# p = prim(results, metric, include=model.uncertainties.keys(), coi="Reliable")
# box = p.find_box()
# box.show_details()
# plt.show()

# use CART to identify the key uncertainties
#c = cart(results, metric, include=model.uncertainties.keys())
#print_tree(c, coi="Reliable")


#print(sa(model, "reliability", policy=policy, method="morris", nsamples=1000, num_levels=4, grid_jump=2))

# # Run model (example)
# Y = Ishigami.evaluate(param_values)
# 
# # Perform analysis
# Si = sobol.analyze(problem, Y, print_to_console=False)
# 
# 
# from SALib.analyze import delta
# 
# 
# 
# print(problem)
# 
# import numpy as np
# 
# X = to_dataframe(model, results, keys=model.uncertainties.keys())
# print(X.values)
# Y = to_dataframe(model, results, keys=["reliability"])
# print(Y.values)
# 
# print(delta.analyze(problem, X.values, Y.values, num_resamples=10, conf_level=0.95, print_to_console=False))


# box1.show_tradeoff().savefig("tradeoff.png")
# fig = box1.show_pairs_scatter()
# fig.set_size_inches((12, 12))
# fig.savefig("scatter.png")

# Basic MORDM analysis where we optimize the model under well-characterized
# uncertainty, then subject each optimal policy to deep uncertain SOWs to
# compute robustness measures


# optimal_policies = optimize(model, NFE=10000)
# 
# 
# SOWs = sample_lhs(model, 100)
# 
# 
# 
# robustness = []
# 
# for policy in optimal_policies:
#     results = evaluate(model, fix(SOWs, policy))
#     robustness.append(mean(check_feasibility(model, results)))
