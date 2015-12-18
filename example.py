import os
import math
import prim
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *

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
model.constraints = [] #[Constraint("reliability >= 0.95")]

# some parameters are levers that we control via our policy
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model.uncertainties = [RealUncertainty("b", 0.1, 0.45),
                       RealUncertainty("q", 2.0, 4.5),
                       RealUncertainty("mean", 0.01, 0.05),
                       RealUncertainty("stdev", 0.001, 0.005),
                       RealUncertainty("delta", 0.93, 0.99)]

if os.path.exists("data.txt"):
    with open("data.txt", "r") as f:
        output = json.load(f)
else:
    output = optimize(model, "NSGAII", 1000)

    with open("data.txt", "w") as f:
        json.dump(output, f)

mpl.rcdefaults()
#sns.set()
#mpl.rcParams["axes.facecolor"] = "white"
cmap = mpl.colors.ListedColormap(sns.color_palette("Blues", 256))
scatter3d(model, output, s="reliability", cmap=cmap)
        
#kdeplot(model, output, x="max_P", y="utility")
#kdeplot(model, output, x="max_P", y="utility", expr=["reliability >= 0.2", "reliability < 0.2"], alpha=0.8)

# pairs(model, output, expr=["reliability >= 0.2", "reliability < 0.2"],
#       palette={"reliability >= 0.2" : sns.color_palette()[0],
#                "reliability < 0.2" : sns.color_palette()[2]})

#joint(model, output, x="max_P", y="utility")

# hist(model, output)

#interact(model, output, "max_P", "utility", "reliability", filled=True)

# construct a specific policy and evaluate it against 1000 states-of-the-world
# policy = {"pollution_limit" : [0.02]*100}
# SOWs = sample_lhs(model, 1000)
# results = evaluate(model, fix(SOWs, policy))
# metric = [1 if v["reliability"] > 0.9 else 0 for v in results]

# use PRIM to identify the key uncertainties if we require reliability > 0.9
# p = prim.Prim(results, metric, exclude=model.levers.keys() + model.responses.keys())
# box = p.find_box()
# box.show_tradeoff()
plt.show()


# df = pd.DataFrame(results)
# 
# for lever in model.levers.keys():
#     df.drop(lever, axis=1, inplace=True)
# for response in model.responses:
#     df.drop(response.name, axis=1, inplace=True)
# prim = Prim(df.to_records(), metric, threshold=0.8, peel_alpha=0.1)
# box1 = prim.find_box()
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
