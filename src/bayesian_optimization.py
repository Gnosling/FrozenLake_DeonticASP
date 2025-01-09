from src.controller import Controller
import datetime

def objective_for_RL_params_1(trial):
    """
    Tests episodes, reverse-q-learning, discount and constant learning-rate
    """
    episodes = trial.suggest_int("episodes", 60, 300)
    # max_steps = trial.suggest_int("max_steps", 9, 20)
    max_steps = 20
    discount = trial.suggest_uniform("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    # initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])
    initialisation = "zero"

    # learning_rate_strategy = trial.suggest_categorical("learning_rate_strategy", ["constant", "linear_decay", "exponential_decay"])
    learning_rate_strategy = "constant"
    if learning_rate_strategy == "linear_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.001, 0.01)
    elif learning_rate_strategy == "exponential_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.0001, 0.001)
    else:
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
        learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A1", config)

    # All trials and their parameters:
    # Trial 216; Value: 0.508; Parameters: {'episodes': 290, 'discount': 0.9831550109680668, 'reversed_q_learning': True, 'learning_rate': 0.09728948807236495};
    # Trial 157; Value: 0.471; Parameters: {'episodes': 292, 'discount': 0.9841257162378517, 'reversed_q_learning': True, 'learning_rate': 0.17626028910729244};
    # Trial 308; Value: 0.47; Parameters: {'episodes': 295, 'discount': 0.8055003189870396, 'reversed_q_learning': True, 'learning_rate': 0.1407057468044647};
    # Trial 167; Value: 0.448; Parameters: {'episodes': 276, 'discount': 0.8048939983986804, 'reversed_q_learning': True, 'learning_rate': 0.1667906347540406};
    # Trial 70; Value: 0.439; Parameters: {'episodes': 270, 'discount': 0.8267467663570353, 'reversed_q_learning': True, 'learning_rate': 0.14462439646471592};

    # TODO: collect result and plot config A1:
    #  config = {"repetitions": 20, "episodes": 300, "max_steps": 20, "evaluation_repetitions": 50,
    #                "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
    #                "learning": {"norm_set": None, "epsilon": None, "initialisation": zero, "reversed_q_learning": true,
    #                             "discount": 0.99, "learning_rate": 0.15, "learning_rate_strategy": constant,
    #                             "learning_decay_rate": None},
    #                "planning": None,
    #                "deontic": {"norm_set": 0, "evaluation_function": 4},
    #                "enforcing": None,
    #                }

def objective_for_RL_params_2(trial):
    """
    Tests learning rate strategies and decays
    """
    episodes = 300
    max_steps = 20
    discount = 0.99
    reversed_q_learning = True
    # initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])
    initialisation = "zero"

    learning_rate_strategy = trial.suggest_categorical("learning_rate_strategy", ["constant", "linear_decay", "exponential_decay"])
    # learning_rate_strategy = "constant"
    if learning_rate_strategy == "linear_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.001, 0.01)
    elif learning_rate_strategy == "exponential_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.0001, 0.001)
    else:
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
        learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A2", config)

    # Trial 147; Value: 0.495; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.37420244005972036};
    # Trial 3; Value: 0.49; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.44026002094667105};
    # Trial 243; Value: 0.485; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.39817010321455726};
    # Trial 64; Value: 0.467; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.19901532270046493};
    # Trial 136; Value: 0.466; Parameters: {'learning_rate_strategy': 'linear_decay' (first: 1.0), 'learning_decay_rate': 0.002599003124983318};


def objective_for_RL_params_3(trial):
    """
    Tests simple initialisation strategies
    """
    episodes = 300
    max_steps = 20
    discount = 0.99
    reversed_q_learning = True
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A3", config)



def bayesian_optimization():
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_for_RL_params_3, n_trials=400, n_jobs=6)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"bayesian_result_from_{current_datetime}.txt"
    with open(file_name, "w") as file:
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=file)

        print_to_file("\n---------------------------------------------------------------------")
        print_to_file("---------------------------------------------------------------------")
        print_to_file("---------------------------------------------------------------------")
        print_to_file(f"Best value: {study.best_value}; Best parameters: {study.best_params}")
        print_to_file("---------------------------------------------------------------------")
        print_to_file("---------------------------------------------------------------------")
        print_to_file("---------------------------------------------------------------------\n")

        sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)[:20]
        print_to_file("\nAll trials and their parameters:")
        for trial in sorted_trials:
            print_to_file(f"Trial {trial.number}; Value: {trial.value}; Parameters: {trial.params};")
