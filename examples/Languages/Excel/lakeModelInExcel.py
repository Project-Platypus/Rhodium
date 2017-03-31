from rhodium import *
from rhodium.excel import ExcelModel

# Provide the Excel file
file = os.path.abspath(os.path.join(os.path.dirname(__file__), "LakeProblem.xlsx"))

with ExcelModel(file) as model:
    # The parameter names must match the R arguments exactly
    model.parameters = [Parameter("pollution_limit", sheet="Inputs", cell="A2:A101"),
                        Parameter("b", sheet="Inputs", cell="B2"),
                        Parameter("q", sheet="Inputs", cell="C2"),
                        Parameter("mean", sheet="Inputs", cell="D2"),
                        Parameter("stdev", sheet="Inputs", cell="E2"),
                        Parameter("delta", sheet="Inputs", cell="F2")]
    
    # List all outputs from the R function, which should return these values either as
    # an unnamed array in the given order (e.g., c(max_P, utility, ...)) or as a named
    # list (e.g., list(max_P=..., utility=..., ...)).
    model.responses = [Response("max_P", Response.MINIMIZE, sheet="Calculations", cell="A8"),
                       Response("utility", Response.MAXIMIZE, sheet="Calculations", cell="A11"),
                       Response("inertia", Response.MAXIMIZE, sheet="Calculations", cell="A14"),
                       Response("reliability", Response.MAXIMIZE, sheet="Calculations", cell="A5")]
    
    # Specify the levers
    model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]
    
    # Optimize the model using Rhodium
    output = optimize(model, "NSGAII", 10000)
    print(output)