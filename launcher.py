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
    # print("START: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # bayesian_optimization("ASP", "8x8_A")

    controller = Controller()
    # experiment = "B1_newstates"
    # controller.run_experiment(experiment)

    experiments = [
        # "B4_greedy", "B4_decay", "B4_newstates", "B4_full",
        # "B5_greedy", "B5_decay", "B5_newstates", "B5_full",
        # "B6_greedy", "B6_decay", "B6_newstates", "B6_full",
        "B7_greedy", "B7_decay", "B7_newstates", "B7_full",
        "B8_greedy", "B8_decay", "B8_newstates", "B8_full",
        "B9_greedy", "B9_decay", "B9_newstates", "B9_full",

    ]

    experiments = [config for config in configs.keys() if config.startswith("C1")]



    for elem in experiments:
        print("START: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        process_config(elem, controller)
        print("FINISHED: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("\n")


    # filtered_configs = [config for config in configs.keys() if config.startswith(experiment)]
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(process_config, config, controller): config for config in filtered_configs}
    #     for future in concurrent.futures.as_completed(futures):
    #         config = futures[future]
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Failed to process {config}: {e.with_traceback()}")

    # controller.plot_compare_of_experiments(["U1","U2","U3","U4","U5","U6","U7"], True, 8)
    # print("FINISHED: " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
