# Introductory Examples

This folder contains several introductory examples.

## example.py

Demonstrates constructing a model in Python and performing various
tasks in Rhodium, including optimization, plotting, sensitivity
analysis, and scenario discovery.

## example_dectorators.py

Demonstrates a shorthand notation for creating models in Python.

## example_parallelization.py

Shows how to parallelize the model evaluations across multiple cores.
This is powered by Platypus' evaluators.  See the module Platypus for
more information on the available parallel evaluators (e.g., MPI).

## sensitivity_analysis.py

Demonstrates the various global sensitivity analysis methods available in
Rhodium (using the SALib library).

## dps_example.py

An example solving the Lake Problem using direct policy search (DPS).

## visualization_with_J3.py

An example visualizing Rhodium's data using the J3 viewer.  J3 is a
cross-platform viewer for high-dimensional data.  To run, you must
install the J3Py module first.