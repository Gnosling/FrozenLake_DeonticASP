from src import *
from src.configs import configs

controller = Controller()

experiment = "T1"

for config in configs.keys():
    if config.startswith(experiment):
        controller.plot_experiment(config)
