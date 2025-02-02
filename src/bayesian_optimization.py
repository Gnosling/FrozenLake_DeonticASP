from src.controller import Controller
import datetime

def objective_for_RL_params_L4_1(trial):
    """
    Tests episodes, reverse-q-learning, discount and constant learning-rate
    """
    episodes = trial.suggest_int("episodes", 60, 300)
    max_steps = 20
    discount = trial.suggest_float("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
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
    return controller.run_experiment(f"A{trial.number}", config)

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
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
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
    return controller.run_experiment(f"A{trial.number}", config)

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
    return controller.run_experiment(f"A{trial.number}", config)

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
    discount = trial.suggest_float("discount", 0.8, 1.0)
    reversed_q_learning = trial.suggest_categorical("reversed_q_learning", [True, False])
    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
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
    return controller.run_experiment(f"A{trial.number}", config)

    # Trial 313; Value: 0.069; Parameters: {'episodes': 276, 'discount': 0.97093809146443, 'reversed_q_learning': True, 'learning_rate': 0.4743161170635867};
    # Trial 153; Value: 0.056; Parameters: {'episodes': 284, 'discount': 0.8235936230792694, 'reversed_q_learning': True, 'learning_rate': 0.06116232548866202};
    # Trial 545; Value: 0.053; Parameters: {'episodes': 279, 'discount': 0.8048033926622332, 'reversed_q_learning': True, 'learning_rate': 0.3690892321793242};
    # Trial 54; Value: 0.052; Parameters: {'episodes': 286, 'discount': 0.8082605763340218, 'reversed_q_learning': True, 'learning_rate': 0.361875475662738};
    # Trial 644; Value: 0.049; Parameters: {'episodes': 286, 'discount': 0.9722123561307507, 'reversed_q_learning': True, 'learning_rate': 0.062071887014868};



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
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
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
    return controller.run_experiment(f"A{trial.number}", config)

    # Trial 687; Value: 0.061; Parameters: {'learning_rate_strategy': 'linear_decay', 'learning_decay_rate': 0.0016759399410973502};
    # Trial 307; Value: 0.059; Parameters: {'learning_rate_strategy': 'constant', 'learning_rate': 0.4710167641035496};
    # Trial 645; Value: 0.053; Parameters: {'learning_rate_strategy': 'linear_decay', 'learning_decay_rate': 0.004562054605694456};
    # Trial 414; Value: 0.049; Parameters: {'learning_rate_strategy': 'linear_decay', 'learning_decay_rate': 0.005158526186860801};
    # Trial 599; Value: 0.047; Parameters: {'learning_rate_strategy': 'linear_decay', 'learning_decay_rate': 0.00431344465456139};


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
    return controller.run_experiment(f"A{trial.number}", config)

    # Trial 306; Value: 0.095; Parameters: {'initialisation': 'safe'};
    # Trial 565; Value: 0.077; Parameters: {'initialisation': 'safe'};
    # Trial 162; Value: 0.075; Parameters: {'initialisation': 'safe'};
    # Trial 301; Value: 0.075; Parameters: {'initialisation': 'safe'};
    # Trial 684; Value: 0.074; Parameters: {'initialisation': 'safe'};

def objective_for_RL_params_L8_1(trial):
    """
    Tests learning-rates
    """
    episodes = 300
    max_steps = 50
    discount = 0.99
    reversed_q_learning = True
    initialisation = trial.suggest_categorical("initialisation", ["safe", "distance"])
    learning_rate_strategy = trial.suggest_categorical("learning_rate_strategy", ["constant", "linear_decay", "exponential_decay"])
    if learning_rate_strategy == "linear_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.001, 0.01)
    elif learning_rate_strategy == "exponential_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.0001, 0.001)
    else:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
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
    return controller.run_experiment(f"A{trial.number}", config)

    # Trial 207; Value: 0.431; Parameters: {'initialisation': 'safe', 'learning_rate_strategy': 'constant', 'learning_rate': 0.4234969090181534};
    # Trial 97; Value: 0.429; Parameters: {'initialisation': 'safe', 'learning_rate_strategy': 'constant', 'learning_rate': 0.35493530380254473};
    # Trial 147; Value: 0.427; Parameters: {'initialisation': 'safe', 'learning_rate_strategy': 'constant', 'learning_rate': 0.432756913857975};
    # Trial 162; Value: 0.42; Parameters: {'initialisation': 'safe', 'learning_rate_strategy': 'constant', 'learning_rate': 0.4389262274624366};
    # Trial 172; Value: 0.418; Parameters: {'initialisation': 'safe', 'learning_rate_strategy': 'constant', 'learning_rate': 0.45048554052623696};


def objective_for_ASP_params_L4_1(trial):
    """
    Tests learning-rates and epsilons
    """
    episodes = 60
    max_steps = 20
    planning_horizon = 10

    initialisation = "distance"
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)

    delta = None
    # planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    planning_strategy = "no_planning"

    learning_rate_strategy = trial.suggest_categorical("learning_rate_strategy", ["constant", "linear_decay", "exponential_decay"])
    if learning_rate_strategy == "linear_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.001, 0.01)
    elif learning_rate_strategy == "exponential_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.0001, 0.001)
    else:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
        learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation, "reversed_q_learning": True,
                            "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy, "planning_horizon": planning_horizon, "reward_set": 2},
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

    # Trial 343; Value: 0.674; Parameters: {'epsilon': 0.02034292352627788, 'learning_rate_strategy': 'constant', 'learning_rate': 0.45788497922633936};
    # Trial 245; Value: 0.671; Parameters: {'epsilon': 0.02599692866575765, 'learning_rate_strategy': 'constant', 'learning_rate': 0.3389808903856023};
    # Trial 321; Value: 0.669; Parameters: {'epsilon': 0.010581231269293893, 'learning_rate_strategy': 'constant', 'learning_rate': 0.43695689297477863};
    # Trial 314; Value: 0.66; Parameters: {'epsilon': 0.014907383996912178, 'learning_rate_strategy': 'constant', 'learning_rate': 0.4352639217095587};
    # Trial 116; Value: 0.659; Parameters: {'epsilon': 0.07264933443963571, 'learning_rate_strategy': 'constant', 'learning_rate': 0.3709629183160711};



def objective_for_ASP_params_L4_2(trial):
    """
    Tests planning strategies and deltas
    """
    episodes = 60
    max_steps = 20
    planning_horizon = 10

    epsilon = 0.02
    delta = None
    planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    if planning_strategy == "delta_greedy_planning":
        delta = trial.suggest_float("delta_greedy", 0.1, 0.9)
    elif planning_strategy == "delta_decaying_planning":
        delta = trial.suggest_loguniform("delta_decaying", 0.00001, 0.0005)

    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
              "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
              "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation,"reversed_q_learning": True,
                           "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                           "learning_decay_rate": learning_decay_rate},
              "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy,
                           "planning_horizon": planning_horizon, "reward_set": 2},
              "deontic": {"norm_set": 0, "evaluation_function": 4},
              "enforcing": None,
              }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

def objective_for_ASP_params_L4_3(trial):
    """
    Tests planning strategies and initialisations
    """
    episodes = 60
    max_steps = 20
    planning_horizon = 20

    epsilon = 0.02
    delta = None
    planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    if planning_strategy == "delta_greedy_planning":
        delta = 0.7 # TODO: set
    elif planning_strategy == "delta_decaying_planning":
        delta = 0.0005 # TODO: set

    initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
              "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
              "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation,"reversed_q_learning": True,
                           "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                           "learning_decay_rate": learning_decay_rate},
              "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy,
                           "planning_horizon": planning_horizon, "reward_set": 2},
              "deontic": {"norm_set": 0, "evaluation_function": 4},
              "enforcing": None,
              }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

def objective_for_ASP_params_L6_1(trial):
    """
    Tests learning-rates and epsilons
    """
    episodes = 150
    max_steps = 25
    planning_horizon = 16

    initialisation = "safe"
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5)

    delta = None
    # planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    planning_strategy = "no_planning"

    learning_rate_strategy = trial.suggest_categorical("learning_rate_strategy", ["constant", "linear_decay", "exponential_decay"])
    if learning_rate_strategy == "linear_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.001, 0.01)
    elif learning_rate_strategy == "exponential_decay":
        learning_rate = 1.0
        learning_decay_rate = trial.suggest_loguniform("learning_decay_rate", 0.0001, 0.001)
    else:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
        learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
               "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation, "reversed_q_learning": True,
                            "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                            "learning_decay_rate": learning_decay_rate},
               "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy, "planning_horizon": planning_horizon, "reward_set": 2},
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": None,
               }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

def objective_for_ASP_params_L6_2(trial):
    """
    Tests planning strategies and deltas
    """
    episodes = 150
    max_steps = 25
    planning_horizon = 16

    epsilon = 0.02 # TODO: set
    delta = None
    planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    if planning_strategy == "delta_greedy_planning":
        delta = trial.suggest_float("delta_greedy", 0.1, 0.9)
    elif planning_strategy == "delta_decaying_planning":
        delta = trial.suggest_loguniform("delta_decaying", 0.00001, 0.0005)

    initialisation = "zero"
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
              "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
              "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation,"reversed_q_learning": True,
                           "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                           "learning_decay_rate": learning_decay_rate},
              "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy, "planning_horizon": planning_horizon, "reward_set": 2},
              "deontic": {"norm_set": 0, "evaluation_function": 4},
              "enforcing": None,
              }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

def objective_for_ASP_params_L6_3(trial):
    """
    Tests planning strategies and initialisations
    """
    episodes = 150
    max_steps = 25
    planning_horizon = 16

    epsilon = 0.02
    delta = None
    planning_strategy = trial.suggest_categorical("planning_strategy", ["no_planning", "full_planning", "plan_for_new_states", "delta_greedy_planning", "delta_decaying_planning"])
    if planning_strategy == "delta_greedy_planning":
        delta = 0.7 # TODO: set
    elif planning_strategy == "delta_decaying_planning":
        delta = 0.0005 # TODO: set

    initialisation = trial.suggest_categorical("initialisation", ["zero", "random", "distance", "safe"])
    learning_rate_strategy = "constant"
    learning_rate = 0.3
    learning_decay_rate = None

    config = {"repetitions": 20, "episodes": episodes, "max_steps": max_steps, "evaluation_repetitions": 50,
              "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
              "learning": {"norm_set": None, "epsilon": epsilon, "initialisation": initialisation,"reversed_q_learning": True,
                           "discount": 0.99, "learning_rate": learning_rate, "learning_rate_strategy": learning_rate_strategy,
                           "learning_decay_rate": learning_decay_rate},
              "planning": {"norm_set": 1, "delta": delta, "strategy": planning_strategy, "planning_horizon": planning_horizon, "reward_set": 2},
              "deontic": {"norm_set": 0, "evaluation_function": 4},
              "enforcing": None,
              }

    controller = Controller()
    controller.disable_storing_and_plottings()
    return controller.run_experiment(f"B{trial.number}", config)

def objective_for_ASP_params_L8_1(trial):
    # TODO: decide
    pass



def bayesian_optimization(category: str, level: str):
    import optuna

    objectives = []
    if category == "RL":
        n_trials = 750
        if level == "4x4_A":
            objectives = [objective_for_RL_params_L4_1, objective_for_RL_params_L4_2, objective_for_RL_params_L4_3]
        elif level == "6x4_A":
            objectives = [objective_for_RL_params_L6_1, objective_for_RL_params_L6_2, objective_for_RL_params_L6_3]
        elif level == "8x8_A":
            objectives = [objective_for_RL_params_L8_1]
    elif category == "ASP":
        n_trials = 100
        if level == "4x4_A":
            objectives = [objective_for_ASP_params_L4_2]

    # TODO: form a baseline of A0? and let it run for some levels to compare with BX later
    # TODO: implement objective-functions for planning strategies (also no planning first for the right epsilon value!), and again with init strat
    #  this affects same levels as for RL
    #  Note: planning uses only the norm reach goal and internal rewards.

    for objective in objectives:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=6)

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
