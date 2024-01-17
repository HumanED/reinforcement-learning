import os
# Model-run number-optional description
run_name = "PPO" + "-" + "1" + "-" + "lunarlander"
models_dir = os.path.join(os.getcwd(),'models')
logs_dir = os.path.join(os.getcwd(),'logs')
if os.path.exists(f"{models_dir}/{run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if os.path.exists(f"{logs_dir}/{run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")