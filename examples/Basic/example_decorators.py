import math
import numpy as np
from scipy.optimize import brentq as root
from rhodium import *
from rhodium.decorators import *
from rhodium.config import RhodiumConfig

@RhodiumModel()
@Responses(Minimize("max_P"), Maximize("utility"), Maximize("inertia"), Maximize("reliability"))
@Constraints("reliability >= 0.95")
def lake_problem(
         pollution_limit = Real(0.0, 0.1, length=100),
         b = Uniform(0.1, 0.45, default_value=0.42),        # decay rate for P in lake (0.42 = irreversible)
         q = Uniform(2.0, 4.5, default_value=2.0),          # recycling exponent
         mean = 0.02,                                       # mean of natural inflows
         stdev = 0.001,                                     # standard deviation of natural inflows
         alpha = 0.4,                                       # utility from pollution
         delta = 0.98,                                      # future utility discount rate
         nsamples = 100):                                   # monte carlo sampling of natural inflows)
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
    inertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (max_P, utility, inertia, reliability)

# Generate 1000 SOWs using a LHS sampling of the uncertainties (b and q)
SOWs = sample_lhs(lake_problem, 1000)

# Apply a fixed pollution_limit policy to each SOW
policy = { "pollution_limit" : [0.01]*100 }
inputs = update(SOWs, policy)

# Evaluate each SOW
setup_cache(file="example.cache")
output = cache("decorators_output", lambda: evaluate(lake_problem, inputs))

# Classify each SOW into two groups: Reliable and Unreliable
output.apply("classification = 'Reliable' if reliability > 0.5 else 'Unreliable'")
print(output)

# Use PRIM to learn how the uncertainties (b and q) affect the classification
p = Prim(output, "classification", coi="Reliable", include=lake_problem.uncertainties.keys())
box = p.find_box()
box.show_tradeoff()
plt.show()