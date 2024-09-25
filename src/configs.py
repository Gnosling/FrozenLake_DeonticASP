configs = {
    "T1": {"repetitions": 5, "episodes": 100, "max_steps": 100,
             "learning": {"discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "initialisation": "safe", "reversed_q_learning": True},
             "frozenlake": {"name": "FrozenLake4x4_T", "slippery": False, "traverser_path": "4x4_T"},
             "planning" : {"policy": "planning", "epsilon": 0.05, "planning_strategy": "plan_for_new_states", "planning_horizon": 9},
             "deontic": {"norm_set": 5, "evaluation_function": 3}
             },
    "T2": {"repetitions": 10, "episodes": 10, "max_steps": 50,
           "discount": 1.0, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0, "reversed_q_learning": False,
           "frozenlake": {"name": "FrozenLake3x3_A", "slippery": False, "tiles": 9, "traverser_path": "3x3_A"},
           "policy": "planning", "epsilon": 0.2, "planning_strategy": "plan_for_new_states", "planning_horizon": 8,
           "norm_set": 1, "evaluation_function": 1
           },
    "T3": {"repetitions": 10, "episodes": 10, "max_steps": 50,
           "discount": 1.0, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake3x3_A", "slippery": False, "tiles": 9, "traverser_path": "3x3_A"},
           "policy": "planning", "epsilon": 0.2, "planning_strategy": "plan_for_new_states", "planning_horizon": 8,
           "norm_set": 1, "evaluation_function": 1
           },

    # A* to test RL-params
    "A0": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1.0, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": False,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A1": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1.0, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    # discounts
    "A2": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A3": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A4": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.75, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A5": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.25, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A6": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.1, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A7": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    # learning_rates
    "A8": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.1, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A9": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.15, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A10": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A11": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.25, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A12": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A13": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.35, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A14": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 0.4, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A15": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 1, "learning_rate": 1, "learning_rate_strategy": "linear_decay", "learning_decay_rate": 0.0001, "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None
           },
    "A16": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "linear_decay", "learning_decay_rate": 0.0005, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A17": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "linear_decay", "learning_decay_rate": 0.001, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A18": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "linear_decay", "learning_decay_rate": 0.005, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A19": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 1, "learning_rate": 1, "learning_rate_strategy": "linear_decay", "learning_decay_rate": 0.01, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },
    "A20": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 1, "learning_rate": 1, "learning_rate_strategy": "exponential_decay", "learning_decay_rate": 0.0005, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },
    "A21": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "exponential_decay", "learning_decay_rate": 0.001, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A22": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "exponential_decay", "learning_decay_rate": 0.005, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A23": {"repetitions": 100, "episodes": 250, "max_steps": 100,
            "discount": 1, "learning_rate": 1, "learning_rate_strategy": "exponential_decay", "learning_decay_rate": 0.01, "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
            "norm_set": None, "evaluation_function": None
            },
    "A24": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 1, "learning_rate": 1, "learning_rate_strategy": "exponential_decay", "learning_decay_rate": 0.05, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },
    # A25 contains Baseline for RL-params
    "A25": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },


    # B* to test policy strategies
    # epsilon tests
    "B1": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.05, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },
    "B2": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.1, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None
             },
    "B3": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.15, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None},
    "B4": {"repetitions": 100, "episodes": 250, "max_steps": 100,
             "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None, "reversed_q_learning": True,
             "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
             "policy": "epsilon_greedy", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
             "norm_set": None, "evaluation_function": None},
    "B5": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.25, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B6": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.3, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B7": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.4, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B8": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.6, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B9": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "epsilon_greedy", "epsilon": 0.8, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    # policy-types test
    "B10": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "greedy", "epsilon": None, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B11": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "exponential_decay", "epsilon": 0.2, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B12": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "exponential_decay", "epsilon": 0.1, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B13": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "exponential_decay", "epsilon": 0.05, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B14": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "exponential_decay", "epsilon": 0.01, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    "B15": {"repetitions": 100, "episodes": 250, "max_steps": 100,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "exponential_decay", "epsilon": 0.005, "planning_strategy": None, "planning_horizon": None,
           "norm_set": None, "evaluation_function": None},
    # Planning policies
    "B16": {"repetitions": 50, "episodes": 100, "max_steps": 150,
           "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
           "reversed_q_learning": True,
           "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
           "policy": "planning", "epsilon": 0.2, "planning_strategy": "full_planning", "planning_horizon": 8,
           "norm_set": 1, "evaluation_function": 1},
    "B17": {"repetitions": 50, "episodes": 100, "max_steps": 150,
            "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
            "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "planning", "epsilon": 0.2, "planning_strategy": "plan_for_new_states", "planning_horizon": 8,
            "norm_set": 1, "evaluation_function": 1},
    "B18": {"repetitions": 50, "episodes": 100, "max_steps": 150,
            "discount": 0.9, "learning_rate": 0.2, "learning_rate_strategy": "constant", "learning_decay_rate": None,
            "reversed_q_learning": True,
            "frozenlake": {"name": "FrozenLake4x4_A", "slippery": True, "tiles": 16, "traverser_path": None},
            "policy": "planning", "epsilon": 0.2, "planning_strategy": "epsilon_planning", "planning_horizon": 8,
            "norm_set": 1, "evaluation_function": 1},

# full_planning
# plan_for_new_states
# epsilon_planning
#

}
