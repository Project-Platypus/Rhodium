# Rhodium

Rhodium is a Python library for robust decision making and exploratory
modeling.  Rhodium is based on the XLRM framework of Robert Lempert et al.  XLRM
stands for:

* **X - Exogeneous uncertainties** are factors that influence the model but
  are outside our control.  For example, in climate models, the average
  temperature in future years is an exogeneous uncertainty.
* **L - Policy levers** are factors that we can control.  For example, a
  national policy could mandate the restriction of carbon emissions below a 
  certain limit.  In Rhodium, we call these "controls".
* **R - Relationships** are embodied in the model and describe how the
  exogeneous uncertainties and policy levers impact the various metrics.
* **M - Metrics** measure the ability of the model to satisfy the decision
  makers' goals and determine how well a policy performs.
  
See our [IPython Notebook](https://gist.github.com/dhadka/a8d7095c98130d8f73bc)
for a demonstration of the current capabilities.