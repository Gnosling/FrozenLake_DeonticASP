do:
create create --name FrozenLake python=3.9
conda activate FrozenLake
conda install swig
conda install -c potassco telingo
pip install -r requirements.txt
% pip install gym[toy_text] %

## ToDo / Status:
- Paper: https://www.overleaf.com/project/666d2ecdd9d2e0c3e9a800ac


- Implement Norms:
  - H: notReachedGoal
  - M: occupiedTraverserTile
  - H: turnedOnTraverserTile (CTD of above)
  - M: stolePresent (took at least one) (conflicting below)
  - M: missedPresents (did not take all available) (conflicting above)
  - L: movedAwayFromGoal (the distance to the goal tile has increased)
  - L: leftSafeArea ('safe' iff not near a hole, only triggers on exit)
  - M: didNotReturnToSafeArea (CTD of above)


- Define settings for evaluation:
  - sum of rewards
  - scaled eval of rewards and violations
  - rewards and violations of level (because the weak constraints work differently in Telingo this simulates that)
  - additionally a 'hard' conflict resolution can be applied to those


- check planning with learning:
  - test combis + RL learning


- let traverser-paths be cyclic, and have a level where it jumps between both paths to block the agent
  - needs new config-param
  - not really needed since path is hardcoded

- implement strategies for exploration for q-table!
- debug what happens if violations and rewards have same level

- Implement plotting and output data:
  - both target and behavior use same q_table, plot only avg target-return over steps
  - extend violations chart with new norms
  - extend plotting for state-visits as heat map
  - have a chart for average number of steps and average number of slips
  - policy map showing the favorite action in each state
  - simply the entire q-table as well (not for plotting)
  - something about exploration (or use that in the heatmap to indicate areas that were not explored?)


- Paradoxes:
  - Ross's paradox
  - Prior's paradox (Paradox of derived obligation)
  - {\AA}qvist’s paradox of Epistemic Obligation
  - Paradox of the Good Samaritan
  - Paradox of free choice
  - Sartre's Dilemma
  - Plato's Dilemma
  - Forrester paradox (The gentle murder paradox)
  - Fence paradox
  - Chisholm's paradox


- Experiments:
  - First on 'crude' frozenlake with better splippery, so everything else deactivated, pick default level (4x4_A # optimum = 0.74)
  - A* for testing RL-params:
    - discount should be high to make the agent use long-term rewards and it's okay because there are mostly rewards negative rewards for any step
    - mention that only value of 1.0 and 0.0 have bad results
    - reverse-q should be better since rewards are only at goal tile
    - random initialisation?? safe initialisation??
  - B* to test policy strategies / classes:
    - test out epsilon -> no signifant value, due to all values lacking at the start
  - Now make extensions to frozenlake
  - C* to test norms simple with CTDs and all evaluations
  - D* to test alternative implementations of norms (ie. forbid neg / oblig pos; deontic vs. factual)
  - E* to test norms inspired to represent concrete paradoxes mentioned in paper (these include also the weird ones)


