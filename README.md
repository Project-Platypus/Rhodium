# Rhodium

Rhodium is a Python library for robust decision making and exploratory
modelling.

## Example

This is a hypothetical example of how Rhodium will operate once developed.
Rhodium is designed based on the XLRM framework of Robert Lempert et al.  XLRM
stand for:

* **X - Exogeneous uncertainties** are factors that influence the model but
  are outside our control.  For example, in climate models, the average
  temperature in future years is an exogeneous uncertainty.
* **L - Policy levers** are factors that we can control.  For example, a
  national policy could mandate the restriction of carbon emmissions below a 
  certain limit.  In Rhodium, we call these "controls".
* **R - Relationships** are embodied in the model and describe how the
  exogeneous uncertainties and policy levers impact the various metrics.
* **M - Metrics** measure the ability of the model to satisfy the decision
  makers' goals and determine how well a policy performs.
  
In Rhodium, we begin with the model.  The model is simply a function in Python.
For example, suppose we are modelling the impacts of pollution on a system.
We could write the function as follows:

```python

    def pollution_model(pollution_limit, b = 0.42, q = 2.0, mean = 0.02,
                        stdev = 0.001, delta = 0.98):
        # evaluate the model
        return [pollution, utility, inertia, reliability]
```

The return value of this function are the metrics, and can include one or more
values.
  
Next, we instantiate a new Rhodium `Model`.  The `Model` class is used
to describe various attributes of our model:

```python

    model = Model(pollution_model)
    model.parameters = [Parameter("pollution_limit"),
                        Parameter("b"),
                        Parameter("q"),
                        Parameter("mean"),
                        Parameter("stdev"),
                        Parameter("delta")]
    model.responses = [Response("Pollution"),
                       Response("Utility", MAXIMIZE),
                       Response("Inertia", MAXIMIZE),
                       Response("Reliability", MAXIMIZE)]
```

Parameters correspond to the inputs to the function, and responses correspond
to the outputs.  Note that the name of the parameter must match the name of the
argument.  Responses work similarly, except we can specify whether the response
is to be minimized (default) or maximized.

There also may be certain conditions we always want to satisfy.  For example,
suppose we require that reliability must be greater than 95%.  We can define
this as a constraint:

```python

    model.constraints = [Constraint("Reliability_Constraint", "Reliability>0.95")]
```

Next, we must ensure the parameters all have valid inputs.  The value of a
parameter can come from several sources depending on the nature of the
parameter.  If the parameter is a policy lever, then we can define it as a
control.  In our example, our policy is restricting the pollution level using
the `pollution_limit` parameter.  Here, we are specifying the limit every
year for the next 100 years, so `pollution_limit` is an array of length
100:

```python

    model.controls = {"pollution_limit" : RealControlArray(0.0, 0.1, length=100)}
```

If the parameter represents an exogeneous uncertainty, then we can define it as
such.  We assume that the uncertain parameter lies between some lower and upper
limit:

```python

    model.uncertainties = {"b" : RealUncertainty(0.1, 0.45),
                           "q" : RealUncertainty(2.0, 4.5),
                           "mean" : RealUncertainty(0.01, 0.05),
                           "stdev" : RealUncertainty(0.001, 0.005),
                           "delta" : RealUncertainty(0.93, 0.99)}
```

Lastly, any other parameters can be given default values.  The default value
can either come from the function definition by assigning the argument a default
value, or define the default value when creating the `Parameter`, e.g.
`Parameter("b", 0.42)`.  Rhodium will display an error if any function
arguments are undefined.

With the model defined in Rhodium, we can then perform a variety of experiments
and analyses:

### Explore Existing Policies

We can explore the impact of existing policies on the model by supplying the
policies directly to the model.  For example, we can create several policies
with different pollution limits:

```python

    policy1 = {"pollution_limit" : [0.01]*100} # strict pollution controls
    policy2 = {"pollution_limit" : [0.06]*100} # moderate pollution controls
    policy3 = {"pollution_limit" : [0.1]*100}  # no pollution controls
```

Since there are uncertainties, we need to assess the effects of these policies
under many possible conditions.  We typically call these "states of the world",
or SOWs for short.  Since we assume that all states are equally likely, we can
use Latin hypercube sampling to generate 100 SOWs:

```python

    SOWs = sample_lhs(model, 100)
```

Next, we can evaluate the impact of each policy across all SOWs:

```python

    result1 = evaluate(model, policy1, SOWs)
    result2 = evaluate(model, policy2, SOWs)
    result3 = evaluate(model, policy3, SOWs)
```

Lastly, we can analyze these results to determine which is the best policy.
Using our constraint of reliability greater than 95%, we can quickly determine
which policy has the highest probability of satisfying our constraint:

```python

    print "Policy 1:", mean(map(is_feasible, result1))
    print "Policy 2:", mean(map(is_feasible, result2))
    print "Policy 3:", mean(map(is_feasible, result3))
```

### Finding Optimal Policies

We can also identify optimal policies for our model.  For this analysis, we
want to determine the optimal values for our controls, such as
`pollution_limit`, that minimize or maximize our metrics.  For a single
metric, there is typically only one optimal policy (or several equally-optimal
policies).  If the model defines more than one metric, then often we encounter
more than one optimal policy.  Each policy is Pareto efficient, performing
better in one or more metrics and worse in others.

In robust optimization, we typically optimize the model accounting for many
SOWs.  In the simplest case, we simply aggregate the average of each metric
within each SOW:

```python

    SOWs = sample_lhs(model, 100)
    robust_optimal_policies = optimize(model, "NSGAII", 10000, SOWs=SOWs, aggregate="average")
```

We could alternatively use `"minimax"` to maximize the minimum value (or
vice versa).  We can also provide a function to compute a more sophisticated
"robustness" measure.

If instead we are performing Robust Decision Making, we would optimize the
model under "well-characterized uncertainty".  For this, we simply use the
default values provided in the model definition.  Afterwards, we subject each
optimal design to the SOWs.  In this manner, we can devise optimal policies
given our best guess of the future SOW, but explore how deviations from our
best guess impact the system.


```python

    optimal_policies = optimize(model, "NSGAII", 10000)
    
    SOWs = sample_lhs(model, 100)
    results = [evaluate(model, policy, SOWs) for policy in optimal_policies]
    feasibility = [mean(map(is_feasible, result)) for result in results]
```

### Identify Vulnerabilities

After evaluating the model against many SOWs, we can attempt to derive simple
rules that lead to infeasible designs.  Here, we can use either Patient Rule
Induction Method (PRIM) or Classification and Regression Trees (CART) to
deduce these rules.

```python

    SOWs = sample_lhs(model, 100)
    results = evaluate(model, policy, SOWs)
    boxes = prim(model, results, is_feasible, threshold_type="<=", threshold=0.5)
```

Here, we use the built-in `is_feasible` function, which assigns True (1) if
the model is feasible within a SOW and False (0) otherwise.  With a
`threshold_type` of `"<="`, we aim to find a subset of the data where
the majority of SOWs with infeasible models.  PRIM identifies lower and upper
bounds for each uncertainty, and hence forms a "box" enclosing the subset.

CART works similarly, except it produces a hierarchical partition of the
data into smaller and smaller subsets.


