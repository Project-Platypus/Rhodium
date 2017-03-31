from rhodium import *
from rhodium.rbridge import RModel

# Provide the R file and function name
model = RModel("lake.R", "lake.eval", RCMD=r"C:\Program Files\R\R-3.2.1\bin\R.exe")

# The parameter names must match the R arguments exactly
model.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

# List all outputs from the R function, which should return these values either as
# an unnamed array in the given order (e.g., c(max_P, utility, ...)) or as a named
# list (e.g., list(max_P=..., utility=..., ...)).
model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

# Specify the levers
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# Optimize the model using Rhodium
output = optimize(model, "NSGAII", 10000)
print(output)
