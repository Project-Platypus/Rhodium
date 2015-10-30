# Rhodium

Rhodium is a Python library for robust decision making and exploratory
modelling.

## Example

This is a hypothetical example of how Rhodium will operate once developed.
Rhodium begins with the model.  The model has parameters (the inputs), 
responses (the outputs), and constraints (conditions that must be satisfied in
order for the model to be feasible).

```python

    model = Model("LakeProblem")
    model.parameters = [Parameter("pollution_limit"),
                        Parameter("b", 0.42),
                        Parameter("q", 2.0),
                        Parameter("mean", 0.02),
                        Parameter("stdev", 0.001),
                        Parameter("delta", 0.98)]
    model.responses = [Response("Pollution"),
                       Response("Utility", MAXIMIZE),
                       Response("Inertia", MAXIMIZE),
                       Response("Reliability", MAXIMIZE)]
    model.constraints = [Constraint("Reliability>95%", ">=0.95")]
```

Both parameters and responses are identified by name.  In addition, parameters
can have default values.  The parameters correspond to the inputs to a function,
such as:

```python

    def lake_model(pollution_limit, b = 0.42, q = 2.0, mean = 0.02,
                   stdev = 0.001, delta = 0.98):
        # evaluate the model
        return [pollution, utility, inertia, reliability]
        
    model.function = lake_model
```

The function can define additional optional arguments that are not enumerated
as parameters.

Next, Rhodium identifies which parameters are controls.  Controls are the model
parameters that can be explicitly controlled.  A set of controls is often
referred to as a *policy*.

```python

    model.controls = {"pollution_limit" : RealControlArray(0.0, 0.1, length=100)}
```

Finally, some of the model's parameters may be uncertain.  That is to say we
are unsure of the actual value of the parameter.  For example, the temperature
could be an uncertainty.  We are confident that the high temperature ranges
between, say, 85 and 105 degrees F during the summer, but we are uncertain of
the actual high temperature on any given day.

```python

    model.uncertainties = {"b" : RealUncertainty(0.1, 0.45),
                           "q" : RealUncertainty(2.0, 4.5),
                           "mean" : RealUncertainty(0.01, 0.05),
                           "stdev" : RealUncertainty(0.001, 0.005),
                           "delta" : RealUncertainty(0.93, 0.99)}
```

Using this model, we can then execute the various steps for robust decision
making or exploratory modelling.  We can identify optimal designs with the
default parameter values:

```python

    optimal_designs = optimize(model, "NSGAII", 10000)
```

or optimize for robust optimal designs, which account for the uncertainties:

```python

    robust_optimal_designs = optimize(model, "NSGAII", 10000, method="robust")
```

For a given optimal design, we can subject it to various conditions based on
the uncertainties.  These conditions are often called "states of the world", or
SOW for short.  Below, we assume the conditions are all equally likely, and use
Latin hypercube sampling to produce 100 hypothetical SOWs:

```python

    SOWs = sample_lhs(model, 100)
    results = [evaluate(model, design, sow) for sow in SOWs]
```


