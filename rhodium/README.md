# Rhodium API reference
## Model

A model object consists of 6 parts

1.	The underlying model, declared with the constructor
2.	model.parameters – the parameters of interest
3.	model.responses – the model responses or outputs
4.	model.constraints – any hard constraints that must be satisfied
5.	model.levers – parameters that decision makers have direct control over
6.	model.uncertainties – parameters that represent exogenous uncertainties

### Rhodium model to be evaluated
`rhodium.Model(function)`

    Creates a model object based off an existing Python function.

#### Parameters:

- function (*function*): A Python function whose arguments match those of desired parameters and returns desired responses.

#### Example

```
>>> model = rhodium.Model(lake_problem)
```
### Model parameters
`rhodium.model.parameters(parameters)`

Defines parameters (inputs) of interest to the Rhodium model. Parameters include all possible levers and uncertainties examined during MORDM analysis.

#### Parameters
- paramters(*list[rhodium.model.Parameter]): a list of Rhodium Parameter objects that defines model paramters

#### Example
```
>>> model.parameters = [rhodium.model.Parameter(“pollution_limit”),
				  rhodium.model.Parameter(“b”),
				  rhodiuim.model.Parameter(“q”),
				  rhodium.model.Parameter(“mean”),
				  rhodium.model.Parameter(“stdev”),
    			  rhodium.model.Parameter(“delta”)]
```
### Model outputs (responses)
`rhodium.model.responses(responses)`
Defines responses (outputs) of interest to the Rhodium model. Responses include performance objectives and any additional information relative to the analysis.

#### Parameters

- responses(*list[rhodium.model.responses]*): a list of Rhodium response objects that define model responses

#### Example

```
>>> model.responses = [rhodium.model.Response(“max_P”, Response.MINIMIZE),
				    rhodium.model.Response(“utility”, Response.MAXIMIZE),
				    rhodiuim.model.Response(“inertia”, Response.MAXIMIZE),
                    rhodium. model.Response(“reliability”, Response.MAXIMIZE)]
```
### Model constraints
`rhodium.model.constraints(constraints)`
Defines hard constraints for the model
#### Parameters
- constraints(*list[rhodium.model.Constraint]*): a list of Rhodium constraint objects that define model constraints
#### Example
```
>>> model.constraints = [Constraint("reliability >= 0.95")]
```
### Model decision variables (levers)
`rhodium.model.levers(levers)`

Defines levers that decision makers control in the Rhodium model

#### Parameters

- levers(*list[rhodium.model.Lever]*): a list of Rhodium lever objects, these can be any type of levers defined in Rhodium

#### Example
```
>>> model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]
```
### Model uncertainties
`rhodium.model.uncertainties(uncertainties)`

Defines exogenous uncertainties and present in the Rhodium model and their plausible ranges.

#### Parameters

- uncertainties(*list[rhodium.model.Uncertainties]*): a list of Rhodium uncertainty objects, these can be any of the defined types of uncertainty in Rhodium.

#### Example

```
>>> model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]
```
## Plotting
### 2D Scatter plot
`rhodium.plot.scatter2d(model, data, x=None, y=None, c=None, s=None, s_range = (10,50), show_colorbar = True, show_legend = False, interactive=False, brush = None, is_class = False, colors=None, **kwargs)`
Returns a figure containing a 2D scatter plot of model responses.

#### Parameters
- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- y ((*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the y-axis
- z (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the z-axis
- c (*rhodium.model.responses.keys*):  The element of the DataSeries object to be plotted as the color
- s (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted as the size
- s_range(*int*, *int*): the maximum and minimum size of points
- show_colorbar(*bool*): boolean specification for colorbar visability
- show_legend(*bool*): boolean specification for legend visability
- brush(*array[rhodium.Brush]*): brushing criteria
- is_class(*bool*): boolean specification for
- colors(*cmap*): Custom colormap specification
-
#### Example
```
>>> fig = scatter2d(model, output, c="reliability")
```
### 3D Scatter plot
`rhodium.plot.scatter3d(model, data, x=None, y=None, z=None, c=None, s=None, s_range = (10,50), show_colorbar = True, show_legend = False, brush = None, pick_handler = None, **kwargs)`

Returns a figure containing a 3D scatter plot of model responses.

#### Parameters
- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the x-axis
- y ((*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the y-axis
- z (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the z-axis
- c (*rhodium.model.responses.keys*):  The element of the DataSeries object to be plotted as the color
- s (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted as the size
- s_range(*int*, *int*): the maximum and minimum size of points
- show_colorbar(*bool*): boolean specification for colorbar visability
- show_legend(*bool*): boolean specification for legend visability
- brush(*array[rhodium.Brush]*): brushing criteria
- pick_handler (): specification for interactive plot

#### Example

```
>>> fig = scatter3d(model, output, c="reliability",
                brush=[Brush("reliability > 0.2"), Brush("reliability <= 0.2")])
```
### Joint plot

`rhodium.plot.joint(model, data, x, y, **kwargs)`

 Returns a Seaborn joint plot of two model responses

#### Parameters

- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the x-axis
- y (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the y-axis

#### Example

```
>>> fig = joint(model, output, x = "reliability", y = "inertia")
```
### Pairwise plot
`rhodium.plot.pairs(model, data, keys = None, brush = None, brush_label = "class", **kwargs)`

Returns a Seaborn pair plot of model responses

#### Parameters
- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- keys (*Rhodium.Model.responses.keys*): The objectives to be plotted, if None, plots all
- brush (*array[rhodium.Brush]*): brushing criteria
- brush_label(*string*): Label of brushed points

#### Example
```
>>> fig = pairs(model, output)
```
### Kernel density estimate plot
`rhodium.plot.kdeplot(model, data, x, y, brush = None, alpha = 1.0, cmap = ["Reds, "Blues", "Oranges", "Greens", "Greys"], **kwargs)`

 Returns a Seaborn kernel density estimate plot of two model responses

#### Parameters

- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*Rhodium.Model.responses.keys*): The objective to be plotted on the x-axis
- y (*Rhodium.Model.responses.keys*): The objective to be plotted on the y-axis
- alpha (*float*): Sets transparency
- cmap (*list(string)*): a list that defines the colormap

#### Example

```
>>> fig = kdeplot(model, output, x="reliability", y="inertia")
```
### Histogram

`rhodium.plot.hist(model, data, keys = None)`

Returns a histogram given model response values

#### Parameters

- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- keys (*Rhodium.Model.responses.keys*): The response to be plotted, if None, plots all

#### Example

```
>>> fig = hist(model, output, keys = "reliability")
```
### Interactive plot
`rhodium.plot.interact(model, data, x, y, z,**kwargs)`

Returns a Seaborn interact plot of given model response values

#### Parameters

- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*Rhodium.Model.responses.keys*): The objective to be plotted on the x-axis
- y (*Rhodium.Model.responses.keys*): The objective to be plotted on the y-axis
- z (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the z-axis

#### Example
```
>>> fig = interact(model, output, x="reliability", y="inertia", z="utility")
```
### 2D contour plot
`rhodium.plot.contour2d(model, data, x=None, y=None, z=None, levels=15, size=100, xlim=None, ylim=None, labels=True, show_colorbar=True, shrink=0.05, method='cubic', **kwargs)`

Returns a seaborn contour plot of given model response values

#### Parameters

- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*Rhodium.Model.responses.keys*): The objective to be plotted on the x-axis
- y (*Rhodium.Model.responses.keys*): The objective to be plotted on the y-axis
- z (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the z-axis
- levels(*int*): number of contours
- size(*int*): mesh size
- xlim (*list[int]*): limits for grid mesh on x-axis
- ylim (*list[int]*): limits for grid mesh on y-axis
- labels (*bool*): boolean to set label visibility
- show_colorbar (*bool*): boolean to set color bar visibility
- shrink (*float*): scaling factor for gird between x and y limits
- method (*string*): interpolation method

#### Example
```
>>> fig = contour2d(model, output, x="reliability", y="inertia")
```
### 3D contour plot
`rhodium.plot.contour3d(model, data, x=None, y=None, z=None, levels=15, size=100, xlim=None, ylim=None, labels=True, show_colorbar=True, shrink=0.05, method='cubic', **kwargs)`

Returns a 3d contour plot of given model response values

#### Parameters
- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- x (*Rhodium.Model.responses.keys*): The objective to be plotted on the x-axis
- y (*Rhodium.Model.responses.keys*): The objective to be plotted on the y-axis
- z (*rhodium.model.responses.keys*): The element of the DataSeries object to be plotted on the z-axis
- levels(*int*): number of contours
- size(*int*): mesh size
- xlim (*list[int]*): limits for grid mesh on x-axis
- ylim (*list[int]*): limits for grid mesh on y-axis
- labels (*bool*): boolean to set label visibility
- show_colorbar (*bool*): boolean to set color bar visibility
- shrink (*float*): scaling factor for gird between x and y limits
- method (*string*): interpolation method

#### Example
```
>>> fig = contour3d(model, output, x="reliability", y="inertia")
```
### Parallel coordinate plot
`rhodium.plot.parallel_coordinates(model, data, c=None, cols=None, ax=None, colors=None, use_columns=False, xticks=None, colormap=None, target="top", brush=None, zorder=None, **kwds)`

Returns a parallel coordinate plot of given model response values

#### Parameters
- model (*rhodium.Model*): A rhodium model which the data comes from
- data (*rhodium.DataSeries*): The data series with model output
- c (*rhodium.model.responses.keys*):  The element of the DataSeries object to be plotted as the color
- cols (*rhodium.model.response.keys*): The responses to be plotted
- colors(*rhodium.model.response.keys*): colors for brushing
- use columns (*bool*): boolean to specify x-axis as columns
- xticks (*list[int]*): spacing for x-axis, must have length equal to number of axes.
- colormap (*cmap*): custom colormap
- targer (*string*): direction of preference on plot (can be "top" or "bottom")
- brush (*array[rhodium.Brush]*): brushing criteria
- zorder (*list[hodium.model.response.keys]*): draw order response should be plotted

#### Example
```
>>> fig = parallel_coordinates(model, output, c="reliability")
```
## Sampling of uncertainties
### Uniform Random Sampling
`rhodium.sampling.sample_uniform(model, nsamples)`

Generate model inputs using Uniform distributions.

Returns a Dataset containing N OrderedDicts, each with D parameter samples, where N is given by `nsamples` and D is equal to the number of uncertain parameters (`model.uncertainties`).

#### Parameters
- model (*rhodium.Model*) - The Rhodium Model object for which to generate sample inputs
- nsamples (*int*) - The number of samples to generate

### Latin Hypercube Sampling
`rhodium.sampling.sample_lhs(model, nsamples)`

Generate model inputs using a Latin Hypercube Sample and the user-defined parameter distributions.

Returns a Dataset containing N OrderedDicts, each with D parameter samples, where N is given by `nsamples` and D is equal to the number of uncertain parameters (`model.uncertainties`).

#### Parameters
- model (*rhodium.Model*) - The Rhodium Model object for which to generate sample inputs
- nsamples (*int*) - The number of samples to generate

## Calculating robustness
### Evaluate the robustness of one or more solutions
`rhodium.robustness.evaluate_robustness(model, policies, SOWs=1000, in_place=True, return_all=False)`

Evaluates the robustness of one or more solutions using four robustness metrics (Regret Type 1, Regret Type 2, Satisficing Type 1, and Satisficing Type 2).

Returns a Dataset containing the objective values achieved by the policies, and the policies' robustness values according to the four metrics.

#### Parameters
- model (*rhodium.Model*) - The Rhodium Model object for which to generate sample inputs
- policies (*rhodium.Dataset*) - Dataset of one or more policies (solutions) to be evaluated
- SOWs (*int*) - Number of states of the world to evaluate robustness over (default 1000)
- in_place (*bool*) - Return robustness values on original *policies* Dataset (default True)
 - return_all (*bool*) - Return sampled states of the world with robustness results (default False)

## Scenario discovery and classification
### Classification And Regression Tree (CART)
`rhodium.classification.Cart(x, y, threshold = None, threshold_type = ">", include = None, exclude = None, **kwargs)`

Generates a decision tree for classification.

#### Parameters
- x - a matrix-like object (pandas.DataFrame, numpy.recarray, etc.), the independent variables
- y - a list-like object, the column name (str), or callable, the dependent variable either provided as a list-like object classifying the data into cases of interest (e.g., False/True), a list-like object storing the raw variable value (in which case a threshold must be given), a string identifying the dependent variable in x, or a function called on each row of x to compute the dependent variable
- threshold (*float*) - threshold for identifying cases of interest
- threshold_type (*str*) - comparison operator used when identifying cases of interest
- include (*list of str*) - the names of variables included in the PRIM analysis
- exclude (*list of str*) - the names of variables excluded from the PRIM analysis
