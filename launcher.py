from src import *
from src.configs import configs
import traceback
import datetime
import concurrent.futures

def process_config(config, controller):
    try:
        controller.run_experiment(config)
    except Exception as exception:
        print(f"ERROR: {exception}")
        traceback.print_exc()


if __name__ == '__main__':
    controller = Controller()
    experiment = "T"
    filtered_configs = [config for config in configs.keys() if config.startswith(experiment)]
    print("START: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_config, config, controller): config for config in filtered_configs}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to process {config}: {e}")

    # controller.plot_compare_of_experiments(["U1","U2","U3","U4","U5","U6","U7"], True, 8)
    print("FINISHED: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
