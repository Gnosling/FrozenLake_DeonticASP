from src import *

controller = Controller()
test_suite = "A"
test_size = 20
# controller.run_experiment("A2")
# TODO: make reverse_q_learning everywhere active in A*
for i in range(1, test_size+1):
    controller.run_experiment(f"{test_suite}{i}")

