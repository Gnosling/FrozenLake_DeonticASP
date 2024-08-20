do:
pip install gym[toy_text]
conda install potassco::telingo
conda install -c potassco telingo

## ToDo / Status:
- Paper: https://www.overleaf.com/project/666d2ecdd9d2e0c3e9a800ac
- Note: add forster / forester paradox

- Implement Norms:
  - (Agent must move towards goal / don't go back / don't visit tile twice / follow planned or given path)
      -  Agent must reach goal is independent of this another norm
  - (If so, then re-return)
  - consider 'applied' paradoxes

- Define settings for evaluation:
  - scaled sum of rewards + violations (maybe split violations with different scales??)

- check planning with learning:
  - test combis + RL learning

- let traverser-paths be cyclic, and have a level where it jumps between both paths to block the agent
  - needs new config-param
- add steal-present with high value, but forbidden norm
- implement norm as forbid and as obligatory? ie O(reachGoal), F(endOnNotGoalTile)

- Implement plotting and output data:
  - both target and behavior use same q_table, plot only avg target-return over steps
  - extend plotting for state-visits as heat map?


- Experiments:
  - First on 'crude' frozenlake with better splippery, so everythin else deactivated, pick default level (4x4_A # optimum = 0.74)
  - A* for testing RL-params:
    - discount should be high to make the agent use long-term rewards and it's okay because there are mostly rewards negative rewards for any step
    - mention that only value of 1.0 and 0.0 have bad results
    - reverse-q should be better since rewards are only at goal tile
    - random initialisation?? safe initialisation??
  - B* to test policy strategies / classes:
    - test out epsilon -> no signifant value, due to all values lacking at the start
  - Now make extensions to frozenlake
  - C* to test norms simple with CTDs and evaluations
  - D* to test alternative implementations of norms
  - E* to test hard norms designed to represent concrete paradoxes mentioned in paper


