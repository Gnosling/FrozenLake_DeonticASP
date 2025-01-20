from src.controller import Controller
import datetime

def objective_for_RL_params_L4_1(trial):
    """
    Tests episodes, reverse-q-learning, discount and constant learning-rate
    """
    episodes = trial.suggest_int("episodes", 60, 300)
    max_steps = 20
    discount = trial.suggest_uniform("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A1", config)

    # Trial 25; Value: 0.496; Parameters: {'episodes': 252, 'discount': 0.8395475115779703, 'reversed_q_learning': True, 'learning_rate': 0.18815556046386858};
    # Trial 132; Value: 0.475; Parameters: {'episodes': 279, 'discount': 0.8466259397519311, 'reversed_q_learning': True, 'learning_rate': 0.20397812380507369};
    # Trial 162; Value: 0.473; Parameters: {'episodes': 249, 'discount': 0.8422325222133493, 'reversed_q_learning': True, 'learning_rate': 0.43197792671208635};
    # Trial 86; Value: 0.468; Parameters: {'episodes': 251, 'discount': 0.9189617029401612, 'reversed_q_learning': True, 'learning_rate': 0.13530255045951342};
    # Trial 602; Value: 0.465; Parameters: {'episodes': 246, 'discount': 0.9357989284024836, 'reversed_q_learning': True, 'learning_rate': 0.11970478397839024};


def objective_for_RL_params_L4_2(trial):
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
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A2", config)

    # Trial 416; Value: 0.493; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.4639995453433418};
    # Trial 251; Value: 0.491; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.3063791469017806};
    # Trial 492; Value: 0.484; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.3063548292055347};
    # Trial 496; Value: 0.474; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.2632327258986996};
    # Trial 748; Value: 0.468; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.47822106030079276};


def objective_for_RL_params_L4_3(trial):
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
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A3", config)

    # Trial 605; Value: 0.7; Parameters: {'initialisation': 'distance'};
    # Trial 177; Value: 0.69; Parameters: {'initialisation': 'distance'};
    # Trial 324; Value: 0.688; Parameters: {'initialisation': 'distance'};
    # Trial 391; Value: 0.685; Parameters: {'initialisation': 'distance'};
    # Trial 743; Value: 0.683; Parameters: {'initialisation': 'distance'};


def objective_for_RL_params_L6_1(trial):
    """
    Tests episodes, reverse-q-learning, discount and constant learning-rate
    """
    episodes = trial.suggest_int("episodes", 100, 300)
    max_steps = 25
    discount = trial.suggest_uniform("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A1", config)



def objective_for_RL_params_L6_2(trial):
    """
    Tests learning rate strategies and decays
    """
    episodes = 300
    max_steps = 25
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
               "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A2", config)


def objective_for_RL_params_L6_3(trial):
    """
    Tests simple initialisation strategies
    """
    episodes = 300
    max_steps = 25
    discount = 0.99
    reversed_q_learning = True
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A3", config)

def objective_for_RL_params_L8_1(trial):
    """
    Tests episodes, reverse-q-learning, discount and constant learning-rate
    """
    episodes = trial.suggest_int("episodes", 100, 300)
    max_steps = 50
    discount = trial.suggest_uniform("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake8x8_A", "traverser_path": "8x8_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A1", config)



def objective_for_RL_params_L8_2(trial):
    """
    Tests learning rate strategies and decays
    """
    episodes = 300
    max_steps = 50
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
               "frozenlake": {"name": "FrozenLake8x8_A", "traverser_path": "8x8_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A2", config)


def objective_for_RL_params_L8_3(trial):
    """
    Tests simple initialisation strategies
    """
    episodes = 300
    max_steps = 50
    discount = 0.99
    reversed_q_learning = True
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake8x8_A", "traverser_path": "8x8_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": None, "initialisation": initialisation, "reversed_q_learning": reversed_q_learning,
                            "discount": discount, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": None,
               "deontic": {"norm_set": 0, "evaluation_function": None},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment("A3", config)

def bayesian_optimization(category: str, level: str):
    import optuna

    objectives = []
    if category == "RL":
        if level == "4x4_A":
            objectives = [objective_for_RL_params_L4_1, objective_for_RL_params_L4_2, objective_for_RL_params_L4_3]
        elif level == "6x4_A":
            objectives = [objective_for_RL_params_L6_1, objective_for_RL_params_L6_2, objective_for_RL_params_L6_3]
        elif level == "8x8_A":
            objectives = [objective_for_RL_params_L8_1, objective_for_RL_params_L8_2, objective_for_RL_params_L8_3]

    # TODO: form a baseline of A0? and let it run for some levels to compare with BX later
    # TODO: implement objective for planning strategies (also no planning first for the right epsilon value!), maybe with more trials or one set foreach level? and again with init strat

    for objective in objectives:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=750, n_jobs=6)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"bayesian_result_of_{str(objective.__name__)}_from_{current_datetime}.txt"
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
            print_to_file("Top 20 trials and their parameters:")
            for trial in sorted_trials:
                print_to_file(f"Trial {trial.number}; Value: {trial.value}; Parameters: {trial.params};")
