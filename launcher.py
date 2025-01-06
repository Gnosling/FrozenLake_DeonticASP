from src import *
from src.configs import configs
import traceback
import concurrent.futures

def process_config(config):
    try:
        controller.plot_experiment(config)
    except Exception as exception:
        print(f"ERROR: {exception}")
        traceback.print_exc()


if __name__ == '__main__':
    controller = Controller()
    experiment = "T"
    filtered_configs = [config for config in configs.keys() if config.startswith(experiment)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_config, config): config for config in filtered_configs}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to process {config}: {e}")


    # controller.plot_compare_of_experiments(["T2", "T3", "T1"], True, 7)
