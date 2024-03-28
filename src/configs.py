configs = {
    "A1": {"reps": 5, "episodes": 3000, "max_steps": 10000,
           "discount": 1.0, "learning_rate": 1.0,
           "frozenlake": {"name": "FrozenLake-v1", "slippery": False, "tiles": 16},
           "policy": "eps_greedy", "epsilon": 0.2,
           "planning": ""
           },
    "A2": {"reps": 3, "episodes": 20, "max_steps": 100,
           "discount": 1.0, "learning_rate": 1.0,
           "frozenlake": {"name": "FrozenLake3x3_A", "slippery": False, "tiles": 9},
           "policy": "eps_greedy", "epsilon": 0.2,
           "planning": ""
           },
}