import os.path

import clingo
import telingo
import os





tel = telingo.TelApp()
# path = os.path.join(os.getcwd(), "levels", "example.lp")
# path = os.path.join(os.getcwd(), "FrozenLake_3x3_A.lp")
path2 = os.path.join(os.getcwd(), "FrozenLake_3x3_A.lp")
path1 = os.path.join(os.getcwd(), "frozenlake_reasoner.lp")
path0 = os.path.join(os.getcwd(), "general_reasoning.lp")
# s = clingo.clingo_main(tel, ['--time-limit=60', '--istop=sat', path])
clingo.clingo_main(tel, ['--time-limit=10', '--istop=sat', path0, path1, path2])

i = 0

