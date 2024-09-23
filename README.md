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
  - M: missedPresents (did not take all) (conflicting above)
  - L: movedAwayFromGoal (the distance to the goal tile has increased)
  - L: leftSafeArea ('safe' iff not near a hole, only triggers on exit)
  - M: didNotReturnToSafeArea (CTD of above)


- Define settings for evaluation:
  - sum of rewards
  - scaled eval of rewards and violations
  - rewards with weak constraints of violations
  - rewards with hard conflict resolution and min of violations
  - rewards with weak constraints of violations and hard conflict resolution


- check planning with learning:
  - test combis + RL learning


- let traverser-paths be cyclic, and have a level where it jumps between both paths to block the agent
  - needs new config-param
  - not really needed since path is hardcoded


- Implement plotting and output data:
  - both target and behavior use same q_table, plot only avg target-return over steps
  - extend violations chart
  - extend plotting for state-visits as heat map?


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


