<img align="left" src="logo.png" />

# Rhodium

[![Tests](https://github.com/Project-Platypus/Rhodium/actions/workflows/tests.yml/badge.svg)](https://github.com/Project-Platypus/Rhodium/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/Rhodium.svg)](https://pypi.python.org/pypi/Rhodium)
[![PyPI](https://img.shields.io/pypi/dm/Rhodium.svg)](https://pypi.python.org/pypi/Rhodium)

Rhodium is an open source Python library for robust decision making (RDM) and multiobjective robust decision
making (MORDM), and exploratory modelling (EM).

## Citation

Please cite the following paper [(PDF)](https://par.nsf.gov/servlets/purl/10314245) if using this project in your own works:

> Hadjimichael A, et al. 2020 Rhodium: Python Library for Many-Objective Robust Decision Making and Exploratory Modeling.
> Journal of Open Research Software, 8: 12. DOI: https://doi.org/10.5334/jors.293

## Installation

To install the latest Rhodium release, run the following command:

```
pip install rhodium
```

To install the latest development version of Rhodium, run the following commands:

```
pip install -U build setuptools
git clone https://github.com/Project-Platypus/Rhodium.git
cd Rhodium
python -m build
```

Rhodium has several optional dependencies that enable additional graphical and analytical capabilities.

1. [GraphViz](http://www.graphviz.org/Download.php) - Required for CART figures (`Cart#show_tree`)
2. `pip install pywin32` - Required to connect to Excel models (`ExcelModel`)
3. `pip install openmdao` - Required to connect to OpenMDAO models (`OpenMDAOModel`)
4. `pip install pyper` - Required to connect to R models (`RModel`)
4. `pip install images2gif` - Required to produce 3D animated GIFs (`animate3d`)
5. [J3Py](https://github.com/Project-Platypus/J3Py) - Interactive 3D visualizations

## Resources

* [Demo IPython Notebook](https://gist.github.com/dhadka/a8d7095c98130d8f73bc)
* [Examples](https://github.com/Project-Platypus/Rhodium/tree/master/examples)

## About

### What is Robust Decision Making?

Robust Decision Making (RDM) is an analytic framework developed by Robert Lempert and his
collaborators at RAND Corporation that helps identify potential robust strategies for a
particular problem, characterize the vulnerabilities of such strategies, and evaluate
trade-offs among them [2].  Multiobjective Robust Decision Making (MORDM)
is an extension of RDM to account for problems with multiple competing performance objectives,
enabling the exploration of performance tradeoffs with respect to robustness
[3, 4].

### What is Rhodium?

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
    return (max_P, utility, inertia, reliability)


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
5. Hadjimichael A, et al. 2020 Rhodium: Python Library for Many-Objective Robust Decision Making and Exploratory Modeling.
   Journal of Open Research Software, 8: 12. DOI: https://doi.org/10.5334/jors.293
