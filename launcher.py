import os

from src import *
from src.bayesian_optimization import bayesian_optimization
from src.configs import configs
import traceback
import datetime
import traceback
import concurrent.futures

def process_config(config, controller):
    try:
        (controller.run_experiment(config))
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Stack trace:")
        traceback.print_exc()

if __name__ == '__main__':
    controller = Controller()

    # TODO: safe init is not the same as safe_area -> check and fix in paper
    # bayesian_optimization("ASP", "8x8_A")

    experiments = [config for config in configs.keys() if config.startswith("E10_guard")]
    print(f"Experiments:\t {experiments}")
    for elem in experiments:
        print("START: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        process_config(elem, controller)
        print("FINISHED: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("\n")

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(process_config, config, controller): config for config in experiments}
    #     for future in concurrent.futures.as_completed(futures):
    #         config = futures[future]
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Failed to process {config}: {e.with_traceback()}")
