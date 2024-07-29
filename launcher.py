from src import *

controller = Controller()
test_suite = "B"
test_size = test_suite_size.get(test_suite)
# controller.run_experiment("A25")
controller.run_experiment("B16")
# controller.run_experiment("B17")
# controller.run_experiment("B18")

# for i in range(1, test_size+1):
#     controller.run_experiment(f"{test_suite}{i}")

