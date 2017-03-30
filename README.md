<img align="left" src="images/logo.png" />

# Rhodium #

Rhodium is an open source Python library for robust decision making (RDM) and multiobjective robust decision
making (MORDM), and exploratory modelling (EM).  Rhodium is still under active development, but it is ready for use.

#### Resources

* [Demo IPython Notebook](https://gist.github.com/dhadka/a8d7095c98130d8f73bc)
* [Installation Instructions](INSTALL.md)
* [Examples](https://github.com/Project-Platypus/Rhodium/tree/master/examples)

#### What is Robust Decision Making?

Robust Decision Making (RDM) is an analytic framework developed by Robert Lempert and his
collaborators at RAND Corporation that helps identify potential robust strategies for a
particular problem, characterize the vulnerabilities of such strategies, and evaluate
trade-offs among them [2].  Multiobjective Robust Decision Making (MORDM)
is an extension of RDM to account for problems with multiple competing performance objectives,
enabling the exploration of performance tradeoffs with respect to robustness
[3, 4].

#### What is Rhodium?

Rhodium is an open source Python library providing methods for RDM and MORDM.  It follows a
declarative design, where you tell Rhodium the actions or analyses you wish to perform and
it determines the necessary calculations.  Rhodium can interface with models written in
a variety of languages, including Python, C and Fortran, R, and Excel.  One begins by
creating a Rhodium model:

```python

from rhodium import *

def lake_problem(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows
    # add body of function
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

model.constraints = [Constraint("reliability >= 0.95")]

model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]
```

A Rhodium model consists of 6 parts:

1. The underlying model (in this case, the Python function `lake_problem`).
2. `model.parameters` - the parameters of interest.
3. `model.responses` - the model responses or outputs.
4. `model.constraints` - any hard constraints that must be satisfied.

5. `model.levers` - parameters that we have direct control over.
6. `model.uncertainties` - parameters that represent exogeneous uncertainties.

Once the Rhodium model is defined, you can then perform any analysis.  For example,
if we want to optimize the model and display the Pareto front:

```python

output = optimize(model, "NSGAII", 10000)
scatter3d(model, output)
plt.show()
```

Check out the [examples](https://github.com/Project-Platypus/Rhodium/tree/master/examples) folder
to see Rhodium in action!

## References

1. Rhodium logo by Tyler Glaude, Creative Commons License, https://thenounproject.com/term/knight/30912/
2. Lempert, R. J., D. G. Groves, S. W. Popper, and S. C. Bankes (2006).  A General, Analytic
   Method for Generating Robust Strategies and Narrative Scenarios.  Management Science, 52(4):514-528.
3. Kasprzyk, J. R., S. Nataraj, P. M. Reed, and R. J. Lempert (2013).  Many objective robust
   decision making for complex environmental systems undergoing change. Environmental
   Modelling & Software, 42:55-71.
4. Hadka, D., Herman, J., Reed, P.M., Keller, K. An Open Source Framework for Many-Objective
   Robust Decision Making. Environmental Modelling & Software, 74:114-129, 2015.
   DOI:10.1016/j.envsoft.2015.07.014. [(View Online)](http://www.sciencedirect.com/science/article/pii/S1364815215300190)
