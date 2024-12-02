from src import *
from src.configs import configs

controller = Controller()

experiment = "U4"

for config in configs.keys():
    if config.startswith(experiment):
        try:
            controller.run_experiment(config)
        except Exception as exception:
            print(f"ERROR: {exception}")
