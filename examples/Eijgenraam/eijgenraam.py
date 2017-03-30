from __future__ import division

import math
import bisect
import numbers
import numpy as np
import matplotlib.pyplot as plt
from rhodium import * 
      
##==============================================================================
## Implement the model described by Eijgenraam et al. (2012)
##------------------------------------------------------------------------------

# Parameters pulled from the paper describing each dike ring
params = ("c", "b", "lam", "alpha", "eta", "zeta", "V0", "P0", "max_Pf")
raw_data = {
    10 : ( 16.6939, 0.6258, 0.0014, 0.033027, 0.320, 0.003774,  1564.9, 0.00044, 1/2000),
    11 : ( 42.6200, 1.7068, 0.0000, 0.032000, 0.320, 0.003469,  1700.1, 0.00117, 1/2000),
    15 : (125.6422, 1.1268, 0.0098, 0.050200, 0.760, 0.003764, 11810.4, 0.00137, 1/2000),
    16 : (324.6287, 2.1304, 0.0100, 0.057400, 0.760, 0.002032, 22656.5, 0.00110, 1/2000),
    22 : (154.4388, 0.9325, 0.0066, 0.070000, 0.620, 0.002893,  9641.1, 0.00055, 1/2000),
    23 : ( 26.4653, 0.5250, 0.0034, 0.053400, 0.800, 0.002031,    61.6, 0.00137, 1/2000),
    24 : ( 71.6923, 1.0750, 0.0059, 0.043900, 1.060, 0.003733,  2706.4, 0.00188, 1/2000),
    35 : ( 49.7384, 0.6888, 0.0088, 0.036000, 1.060, 0.004105,  4534.7, 0.00196, 1/2000),
    38 : ( 24.3404, 0.7000, 0.0040, 0.025321, 0.412, 0.004153,  3062.6, 0.00171, 1/1250),
    41 : ( 58.8110, 0.9250, 0.0033, 0.025321, 0.422, 0.002749, 10013.1, 0.00171, 1/1250),
    42 : ( 21.8254, 0.4625, 0.0019, 0.026194, 0.442, 0.001241,  1090.8, 0.00171, 1/1250),
    43 : (340.5081, 4.2975, 0.0043, 0.025321, 0.448, 0.002043, 19767.6, 0.00171, 1/1250),
    44 : ( 24.0977, 0.7300, 0.0054, 0.031651, 0.316, 0.003485, 37596.3, 0.00033, 1/1250),
    45 : (  3.4375, 0.1375, 0.0069, 0.033027, 0.320, 0.002397, 10421.2, 0.00016, 1/1250),
    47 : (  8.7813, 0.3513, 0.0026, 0.029000, 0.358, 0.003257,  1369.0, 0.00171, 1/1250),
    48 : ( 35.6250, 1.4250, 0.0063, 0.023019, 0.496, 0.003076,  7046.4, 0.00171, 1/1250),
    49 : ( 20.0000, 0.8000, 0.0046, 0.034529, 0.304, 0.003744,   823.3, 0.00171, 1/1250),
    50 : (  8.1250, 0.3250, 0.0000, 0.033027, 0.320, 0.004033,  2118.5, 0.00171, 1/1250),
    51 : ( 15.0000, 0.6000, 0.0071, 0.036173, 0.294, 0.004315,   570.4, 0.00171, 1/1250),
    52 : ( 49.2200, 1.6075, 0.0047, 0.036173, 0.304, 0.001716,  4025.6, 0.00171, 1/1250),
    53 : ( 69.4565, 1.1625, 0.0028, 0.031651, 0.336, 0.002700,  9819.5, 0.00171, 1/1250)}
data = {i : {k : v for k, v in zip(params, raw_data[i])} for i in six.iterkeys(raw_data)}

# Set the ring we are analyzing
ring = 15
max_failure_probability = data[ring]["max_Pf"]
   
# Compute the investment cost to increase the dike height
def exponential_investment_cost(u,     # increase in dike height
                                h0,    # original height of the dike
                                c,     # constant from Table 1
                                b,     # constant from Table 1
                                lam):  # constant from Table 1
    if u == 0:
        return 0
    else:
        return (c + b*u)*math.exp(lam*(h0+u))
    
# The Python function implementing the model
def eijgenraam(Xs,                          # list of dike heightenings
               Ts,                          # time of dike heightenings
               T = 300,                     # planning horizon
               P0 = data[ring]["P0"],       # constant from Table 1
               V0 = data[ring]["V0"],       # constant from Table 1
               alpha = data[ring]["alpha"], # constant from Table 1
               delta = 0.04,                # discount rate, mentioned in Section 2.2
               eta = data[ring]["eta"],     # constant from Table 1
               gamma = 0.035,               # paper says this is taken from government report, but no indication of actual value
               rho = 0.015,                 # risk-free rate, mentioned in Section 2.2
               zeta = data[ring]["zeta"],   # constant from Table 1
               c = data[ring]["c"],         # constant from Table 1
               b = data[ring]["b"],         # constant from Table 1
               lam = data[ring]["lam"]):    # constant from Table 1
    Ts = [int(Ts[i] + sum(Ts[:i])) for i in range(len(Ts)) if Ts[i] + sum(Ts[:i]) < T]
    Xs = Xs[:len(Ts)]
    
    if len(Ts) == 0:
        Ts = [0]
        Xs = [0]
        
    if Ts[0] > 0:
        Ts.insert(0, 0)
        Xs.insert(0, 0)
    
    S0 = P0*V0
    beta = alpha*eta + gamma - rho
    theta = alpha - zeta
    
    # calculate investment
    investment = 0
    
    for i in range(len(Xs)):
        step_cost = exponential_investment_cost(Xs[i], 0 if i==0 else sum(Xs[:i]), c, b, lam)
        step_discount = math.exp(-delta*Ts[i])
        investment += step_cost * step_discount
    
    # calculate expected losses
    losses = 0
    
    for i in range(len(Xs)-1):
        losses += math.exp(-theta*sum(Xs[:(i+1)]))*(math.exp((beta - delta)*Ts[i+1]) - math.exp((beta - delta)*Ts[i]))
        
    if Ts[-1] < T:
        losses += math.exp(-theta*sum(Xs))*(math.exp((beta - delta)*T) - math.exp((beta - delta)*Ts[-1]))

    losses = losses * S0 / (beta - delta)
    
    # salvage term
    losses += S0*math.exp(beta*T)*math.exp(-theta*sum(Xs))*math.exp(-delta*T) / delta
    
    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return sum(Xs)
        else:
            return sum(Xs[:bisect.bisect_right(Ts, t)])
    
    failure_probability = [P0*np.exp(alpha*eta*t)*np.exp(-alpha*find_height(t)) for t in range(T+1)]
    total_failure = 1 - functools.reduce(operator.mul, [1 - p for p in failure_probability], 1)
    mean_failure = sum(failure_probability) / (T+1)
    max_failure = max(failure_probability)
    
    return (investment, losses, investment+losses, total_failure, mean_failure, max_failure)

# Generate a plot showing the dike heightenings and failure probability over time
def plot_details(
               Xs,                          # list of dike heightenings
               Ts,                          # time of dike heightenings
               T = 300,                     # planning horizon
               P0 = data[ring]["P0"],       # constant from Table 1
               alpha = data[ring]["alpha"], # constant from Table 1
               eta = data[ring]["eta"],     # constant from Table 1
               plot_args = {}):   
    Ts = [int(Ts[i] + sum(Ts[:i])) for i in range(len(Ts)) if Ts[i] + sum(Ts[:i]) < T]
    Xs = Xs[:len(Ts)]
    
    if len(Ts) == 0:
        Ts = [0]
        Xs = [0]
        
    # convert inputs to numpy arrays
    P0 = np.asarray([P0]) if isinstance(P0, numbers.Number) else np.asarray(P0)
    alpha = np.asarray([alpha]) if isinstance(alpha, numbers.Number) else np.asarray(alpha)
    eta = np.asarray([eta]) if isinstance(eta, numbers.Number) else np.asarray(eta)
    n = max([x.shape[0] for x in [P0, alpha, eta]])
    
    # compute the failure probability
    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return sum(Xs)
        else:
            return sum(Xs[:bisect.bisect_right(Ts, t)])
              
    failure_probability = np.empty((n, T+1))
    
    for t in range(T+1):
        failure_probability[:,t] = P0*np.exp(alpha*eta*t)*np.exp(-alpha*find_height(t))

    # generate the plot
    fig = plt.figure()

    for i in range(n):
        plt.plot(range(T+1), failure_probability[i,:], 'b-', **plot_args)
            
    plt.plot(range(T+1), [max_failure_probability]*(T+1), 'r--')
    
    for i in range(len(Ts)):
        if Ts[i] == 0:
            plt.text(0, np.max(failure_probability[:,Ts[0]])/2, str(round(Xs[i], 1)) + " cm", ha='left', va='center')
        else:
            plt.text(Ts[i], (np.max(failure_probability[:,Ts[i]-1])+np.max(failure_probability[:,Ts[i]]))/2, str(round(Xs[i], 1)) + " cm", ha='left', va='center')
    
    if n == 1:
        plt.legend(["Failure Probability", "Current Safety Standard"])
    else:
        plt.ylim([0, 0.001])
    
    plt.xlabel("Time (years)")
    plt.ylabel("Failure Probability")
    plt.show()

##==============================================================================
## Create the model in Rhodium
##------------------------------------------------------------------------------
model = Model(eijgenraam)
 
model.parameters = [Parameter("Xs"),
                    Parameter("Ts"),
                    Parameter("T"),
                    Parameter("P0"),
                    Parameter("V0"),
                    Parameter("alpha"),
                    Parameter("delta"),
                    Parameter("eta"),
                    Parameter("gamma"),
                    Parameter("rho"),
                    Parameter("zeta"),
                    Parameter("c"),
                    Parameter("b"),
                    Parameter("lam")]

# For this example, we replicates the results from the paper and only minimize
# total cost.  See eijgenraam_mordm.py for a multiobjective example.
model.responses = [Response("TotalInvestment", Response.INFO),
                   Response("TotalLoss", Response.INFO),
                   Response("TotalCost", Response.MINIMIZE),
                   Response("TotalFailureProb", Response.INFO),
                   Response("AvgFailureProb", Response.INFO),
                   Response("MaxFailureProb", Response.INFO)]

# Each height/time pair defines the dike height increase, Xs[i], after waiting
# some number of years, Ts[i], from the previous dike heightening.  If
# sum(Ts[0..i]) exceeds the planning horizon, T, then any remaining heightenings
# are ignored.
model.levers = [RealLever("Xs", 0, 500, length=6),
                RealLever("Ts", 0, 300, length=6)]


##==============================================================================
## Setup cache
##------------------------------------------------------------------------------
setup_cache(file="eijgenraam_ring%d.cache" % ring)

##==============================================================================
## Optimize the model (caching result to avoid recomputing each time)
##------------------------------------------------------------------------------
policies = cache("policies", lambda: optimize(model, "NSGAII", 100000))

##==============================================================================
## Analyze the results
##------------------------------------------------------------------------------
policy = policies[0]

# display the policy
plot_details(policy["Xs"], policy["Ts"])

# see how the policy works given uncertainty in a few parameters
model.uncertainties = [LogNormalUncertainty("P0", 0.00137, 0.25),
                       NormalUncertainty("alpha", 0.0502, 0.01),
                       LogNormalUncertainty("eta", 0.76, 0.1)]
     
SOWs = sample_lhs(model, 1000)
plot_details(policy["Xs"], policy["Ts"], P0=SOWs["P0"], alpha=SOWs["alpha"], eta=SOWs["eta"], plot_args={"alpha" : 0.02})
