import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *

# Configure the lake model
def lake_problem(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(pollution_limit)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    intertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (max_P, utility, intertia, reliability)

model = Model(lake_problem)

model.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]

# Rhodium performs sensitivity analysis on the uncertainties with respect to a response.
# For example, here we are interested in the sensitivity of the reliability response to
# the uncertain parameters.  Here, we'll perform the analysis on a single policy.
#
# It's important to understand the underlying assumptions of each method.  Some methods
# may require fewer or more samples to produce accurate results.  Additionally, different
# methods report sensitivities in different ways; typically we recommend using the ranking
# of sensitivities rather than the relative value to measure the importance of different
# uncertainties.
#
# Rhodium currently supports:
#
#    "fast" - Fourier Amplitude Sensitivity Test,
#    "morris" - Method of Morris,
#    "delta" - Borgonovo's Delta Moment-Independent Method,
#    "sobol" - Sobol Sensitivity Analysis,
#    "ff" - Fractional Factorial Sensitivity Analysis, and
#    "dgsm" - Derivative-based Global Sensitivity Measure.
#
# Several of these methods are demonstrated below.
#
# In addition to the text output, you can call plot() on the results object to visualize
# the data in a plot.  Sobol's method supports a special radial plot accessible by calling
# plot_sobol().

policy = { "pollution_limit" : [0.075]*100 }

fast_results = sa(model, "reliability", policy=policy, method="fast", nsamples=1000)
print("Fourier Amplitude Sensitivity Test (FAST)")
print("------------------------------------------------------------")
print(fast_results)
print()
 
delta_results = sa(model, "reliability", policy=policy, method="delta", nsamples=1000)
print("Borgonovo's Delta Moment-Independent Measure")
print("------------------------------------------------------------")
print(delta_results)
print()
 
sobol_results = sa(model, "reliability", policy=policy, method="sobol", nsamples=10000)
print("Sobol Sensitivity Analysis")
print("------------------------------------------------------------")
print(sobol_results)
print()

morris_results = sa(model, "reliability", policy=policy, method="morris", nsamples=1000, num_levels=4, grid_jump=2)
print("Method of Morris")
print("------------------------------------------------------------")
print(morris_results)
print()
