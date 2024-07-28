from src import *

controller = Controller()
test_suite = "A"
test_size = 25
# controller.run_experiment("A25")
# controller.run_experiment("A20")

for i in range(0, test_size+1):
    controller.run_experiment(f"{test_suite}{i}")

