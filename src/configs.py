configs = {

    # "T0": {"repetitions": 3, "episodes": 20, "max_steps": 30,
    #        "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": None, "slippery": True},
    #        "learning": {"norm_set": 7, "epsilon": 0.3, "initialisation": "zero | random | distance | safe | state_function | state_action_penalty", "reversed_q_learning": True, "discount": 0.95, "learning_rate": 0.6, "learning_rate_strategy": "constant", "exponential_decay", "linear_decay", "learning_decay_rate": 0.02},
    #        "planning": {"norm_set": 7, "delta": 0.5, "strategy": "no_planning | full_planning | plan_for_new_states | delta_greedy_planning | delta_decaying_planning", "planning_horizon": 14, "reward_set" : 1},
    #        "deontic": {"norm_set": 8, "evaluation_function": 3},
    #        "enforcing": {"norm_set": 6, "strategy": "guardrail | fixing | optimal_reward_shaping | full_reward_shaping", "phase": "during_training | after_training", "enforcing_horizon": [3,6] (no use in guardral; in fixing is list [len of checked path; len of fixed path]; in reward-shaping defines number of shaping steps)},
    #        },

    "T1": {"repetitions": 50, "episodes": 65, "max_steps": 15, "evaluation_repetitions": 100,
           "frozenlake": {"name": "FrozenLake3x3_A", "traverser_path": "3x3_B", "slippery": True},
           "learning": {"norm_set": None, "epsilon": None, "initialisation": "distance", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": None,
           "deontic": {"norm_set": 0, "evaluation_function": None},
           "enforcing": {"norm_set": 3, "strategy": "guardrail", "phase": "after_training", "enforcing_horizon": None}
           },

    "T2": {"repetitions": 2, "episodes": 30, "max_steps": 15, "evaluation_repetitions": 20,
               "frozenlake": {"name": "FrozenLake3x3_A", "traverser_path": "3x3_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
               "planning": {"norm_set": 3, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 8, "reward_set": 2},
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               "enforcing": {"norm_set": 3, "strategy": "optimal_reward_shaping", "phase": "after_training", "enforcing_horizon": [15]},
               },

    "T3": {"repetitions": 1, "episodes": 60, "max_steps": 15, "evaluation_repetitions": 20,
               "frozenlake": {"name": "FrozenLake3x3_A", "traverser_path": "3x3_A", "slippery": True},
               "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
               "planning": {"norm_set": 3, "delta": 0.8, "strategy": "delta_greedy_planning", "planning_horizon": 8, "reward_set": 2},
               "deontic": {"norm_set": 0, "evaluation_function": 4},
               },

    "ww4": {"repetitions": 3, "episodes": 15, "max_steps": 40,
           "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": False},
           "learning": {"epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.95, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None},
           "planning": {"delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 12},
           "deontic": {"norm_set": 3, "evaluation_function": 3},
           "enforcing": {"norm_set": 6, "strategy": "full_reward_shaping", "phase": "after_training", "enforcing_horizon": 4},
           },



    "U4_1": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 10, "strategy": "guardrail", "phase": "after_training", "enforcing_horizon": None},
           },
    "U6_1": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 11, "strategy": "guardrail", "phase": "after_training", "enforcing_horizon": None},
           },

    "U4_2": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 10, "strategy": "fixing", "phase": "after_training", "enforcing_horizon": [4,9]},
           },
    "U6_2": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 11, "strategy": "fixing", "phase": "after_training", "enforcing_horizon": [7,14]},
           },

    "U4_3": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 10, "strategy": "optimal_reward_shaping", "phase": "after_training", "enforcing_horizon": [60]},
           },
    "U6_3": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 11, "strategy": "optimal_reward_shaping", "phase": "after_training", "enforcing_horizon": [75]},
           },


    "U4_4": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 10, "strategy": "full_reward_shaping", "phase": "after_training", "enforcing_horizon": [60]},
           },
    "U6_4": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": {"norm_set": 11, "strategy": "full_reward_shaping", "phase": "after_training", "enforcing_horizon": [75]},
           },

    "U4_5": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": 10, "epsilon": 0.3, "initialisation": "state_function", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },
    "U6_5": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": 11, "epsilon": 0.3, "initialisation": "state_function", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },

    "U4_6": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": 10, "epsilon": 0.3, "initialisation": "state_action_penalty", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },
    "U6_6": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": 11, "epsilon": 0.3, "initialisation": "state_action_penalty", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },
    "U4_7": {"repetitions": 20, "episodes": 60, "max_steps": 20, "evaluation_repetitions": 20,
           "frozenlake": {"name": "FrozenLake4x4_B", "traverser_path": "4x4_B", "slippery": True},
           "learning": {"norm_set": 10, "epsilon": 0.3, "initialisation": "distance", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 8, "delta": 0.5, "strategy": "delta_greedy_planning", "planning_horizon": 9, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },
    "U6_7": {"repetitions": 20, "episodes": 100, "max_steps": 30,
           "frozenlake": {"name": "FrozenLake6x4_A", "traverser_path": "6x4_A", "slippery": True},
           "learning": {"norm_set": 11, "epsilon": 0.3, "initialisation": "distance", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": 0.02},
           "planning": {"norm_set": 9, "delta": 0.75, "strategy": "delta_greedy_planning", "planning_horizon": 14, "reward_set" : 2},
           "deontic": {"norm_set": 0, "evaluation_function": 4},
           "enforcing": None,
           },

    # A* to test RL-params
    # A0 is final baseline

    # "A0": {"repetitions": 20, "episodes": 80, "max_steps": 50,
    #        "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": None, "slippery": True},
    #        "learning": {"norm_set": None, "epsilon": 0.3, "initialisation": "zero | random | distance | safe | state_function | state_action_penalty", "reversed_q_learning": True, "discount": 0.95, "learning_rate": 0.6, "learning_rate_strategy": "constant | linear_decay | exponential_decay", "learning_decay_rate": 0.02},
    #        "planning": None,
    #        "deontic": {"norm_set": 8, "evaluation_function": None},
    #        "enforcing": None
    #        },

    "A1": {"repetitions": 100, "episodes": 300, "max_steps": 20, "evaluation_repetitions": 100,
           "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": None, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.15, "learning_rate_strategy": "constant", "learning_decay_rate": None},
           "planning": None,
           "deontic": {"norm_set": 0, "evaluation_function": None},
           "enforcing": None,
           },

    "A2": {"repetitions": 100, "episodes": 300, "max_steps": 20, "evaluation_repetitions": 100,
           "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": None, "initialisation": "zero", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.35, "learning_rate_strategy": "constant", "learning_decay_rate": None},
           "planning": None,
           "deontic": {"norm_set": 0, "evaluation_function": None},
           "enforcing": None,
           },

    "A3": {"repetitions": 100, "episodes": 300, "max_steps": 20, "evaluation_repetitions": 100,
           "frozenlake": {"name": "FrozenLake4x4_A", "traverser_path": "4x4_A", "slippery": True},
           "learning": {"norm_set": None, "epsilon": None, "initialisation": "distance", "reversed_q_learning": True, "discount": 0.99, "learning_rate": 0.3, "learning_rate_strategy": "constant", "learning_decay_rate": None},
           "planning": None,
           "deontic": {"norm_set": 0, "evaluation_function": None},
           "enforcing": None,
           },

    # TODO: repeat bayesian experiments for two other levels, define which (also repeat current one) [4x4_A, 6x4_A, 8x8_A]
    # TODO: define 'baseline' for majority of levels afterwards (we don't consider norms or presents, only traverser when cracked),
    #  level without norms: 3x3_A, 4x4_A, 6x4_A, 6x4_B, 7x4_A, 7x4_B, 7x4_C, 8x8_A
    # TODO: text 7x4 + 8x8 -levels!!


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
