import numpy as np
from scipy.optimize import brentq as root
from rhodium.model import *

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
        X[0] = 0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in xrange(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/nsamples
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    intertia = np.sum(np.diff(decisions) > -0.02)/(nvars-1)
    
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
model.levers = {"pollution_limit" : RealLever(0.0, 0.1, length=100)}

# some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model.uncertainties = {"b" : RealUncertainty(0.1, 0.45),
                       "q" : RealUncertainty(2.0, 4.5),
                       "mean" : RealUncertainty(0.01, 0.05),
                       "stdev" : RealUncertainty(0.001, 0.005),
                       "delta" : RealUncertainty(0.93, 0.99)}

# compare explicit policies
policy1 = {"pollution_limit" : [0.0]*100}
policy2 = {"pollution_limit" : [0.1]*100}

print evaluate(model, policy1)
print evaluate(model, policy2)

# evaluate model in randomly-generated SOWs (the fix() function assigns the
# same policy within each SOW)
SOWs = sample_lhs(model, 100)
for SOW in SOWs:
    print evaluate(model, fix(SOW, policy1))

# evaluate a policy against many SOWs and compute the percentage of SOWs that
# are feasible
results = evaluate(model, fix(SOWs, policy1))
print mean(check_feasibility(model, results))

# Basic MORDM analysis where we optimize the model under well-characterized
# uncertainty, then subject each optimal policy to deep uncertain SOWs to
# compute robustness measures
optimal_policies = optimize(model, NFE=1000)
SOWs = sample_lhs(model, 100)
robustness = []

for i, policy in enumerate(optimal_policies):
    print "Evaluating design", i
    results = evaluate(model, fix(SOWs, policy))
    robustness.append(mean(check_feasibility(model, results)))
