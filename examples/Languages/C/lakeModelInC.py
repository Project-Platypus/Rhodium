from rhodium import *
from rhodium.ffi import NativeModel

# Provide the DLL/SO file and the function name.  The appropriate extension,
# such as .dll or .so, will be automatically added.
model = NativeModel("lake", "lake_problem")

# List the inputs.  The order matters!
model.parameters = [Parameter("pollution_limit", type="double*"),
                    Parameter("b", default_value=0.42, type="double"),
                    Parameter("q", default_value=2, type="double"),
                    Parameter("mean", default_value=0.02, type="double"),
                    Parameter("stdev", default_value=0.001, type="double"),
                    Parameter("delta", default_value=0.98, type="double")]

# List all outputs.  We use asarg=True to handle the outputs as arguments to the C
# function.
model.responses = [Response("max_P", Response.MINIMIZE, type="double", asarg=True),
                   Response("utility", Response.MAXIMIZE, type="double", asarg=True),
                   Response("inertia", Response.MAXIMIZE, type="double", asarg=True),
                   Response("reliability", Response.MAXIMIZE, type="double", asarg=True)]

# Specify the levers
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# Optimize the model using Rhodium
output = optimize(model, "NSGAII", 10000)
print(output)
