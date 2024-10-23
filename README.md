first time setup:
create create --name FrozenLake python=3.9
conda activate FrozenLake
conda install swig
conda install -c potassco telingo
pip install -r requirements.txt
% pip install gym[toy_text] %

## ToDo / Status:
- Paper: https://www.overleaf.com/project/666d2ecdd9d2e0c3e9a800ac


- Norms:
  - H: notReachedGoal
  - M: occupiedTraverserTile
  - H: turnedOnTraverserTile (CTD of above)
  - M: stolePresent (took at least one) (conflicting below) (Agent always picks up first)
  - M: missedPresents (did not take all available) (conflicting above)
  - L: movedAwayFromGoal (the distance to the goal tile has increased)
  - L: leftSafeArea ('safe' iff not near a hole, only triggers on exit)
  - M: didNotReturnToSafeArea (CTD of above)


- Evaluation:
  - sum of rewards
  - scaled eval of rewards and violations
  - rewards and violations of level (because the weak constraints work differently in Telingo this simulates that)
  - additionally a 'hard' conflict resolution can be applied to those

---------------

- check planning with learning:
  - test combis + RL learning


- non-deterministic policy --> not needed / beneficial


### policy can be enforced:
The enforcement can happen during the training (ie. behavior policy) or afterwards (ie. the final target-policies).
During training if exploration is triggered no enforce-ment is applied.

- --> guardrail (maybe utilizes something from davide?, at least emery's approach):
  - reduces action selections before policy (only on current state) beforehand (can be checked in python-code)
  - might restrict optimal solutions and exploration, but simple and hopefully effective
- --> fixing (sebastians approach):
  - the next enforcing-horizon actions proposed by policy are analysed by ASP-planner for norm violations, no policy changed though
  - -> the horizon must be high enough to compute a path to the end of the level, if reachedGoal-norm is used
  - the current act(move(X)) of the path must be inserted dynamically and checked for violations (also non-deterministic ones?), if any occured then activate normal planning
  - need special ASP-checker of norms, how does this relate with the normal planner?
  - compared to others high computational effort, but most flexible and best monitoring 
- --> reward_shaping (paper-one):
  - the rewards use penalties if violations occur, up to the last enforcing-horizon states
  - (maybe we can use alteration of the policy still, as maybe the fourth mode?)
  - might worsen policy also is in the 'real' testing phase, but could improve, check paper to see how to do that
  - --> as long as function is potential-based, there is no drop in optimal policy


- should the traverser be part of the state-info? --> yes --> define state representation in overleaf (maybe use both?)
- --> this is pair of states (also include presents?-> yes)
- --> state space should be fine since it's: Tiles×Tiles×(Possible Configurations of Presents), at worst T^2 x 2^T x actions, but there aren't that many presents there
- --> use at most 3 presents, maybe less then it's fine
- implement strategies for exploration for q-table!
- implement new plannig strategy for comparing actions and pick better?
- implement distance based init of table
- debug what happens if violations and rewards have same level


### Implement plotting and output data:
  - both target and behavior use same q_table, plot only avg target-return over steps
  - extend violations chart with new norms
  - extend plotting for state-visits as heat map
  - have a chart for average number of steps and average number of slips
  - policy map showing the favorite action in each state
  - simply the entire q-table as well (not for plotting)
  - something about exploration (or use that in the heatmap to indicate areas that were not explored?)

---------------

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

---------------

- Experiments:
  - First on 'crude' frozenlake with better splippery, so everything else deactivated, pick default level (4x4_A # optimum = 0.74)
  - A* for testing RL-params:
    - discount should be high to make the agent use long-term rewards and it's okay because there are mostly rewards negative rewards for any step
    - mention that only value of 1.0 and 0.0 have bad results
    - reverse-q should be better since rewards are only at goal tile
    - random initialisation?? safe initialisation??
  - B* to test policy strategies / classes:
    - test out epsilon -> no signifant value, due to all values lacking at the start
    - test out different starting tiles on same level? with same policy?
  - Now make extensions to frozenlake
  - C* to test norms simple with CTDs and all evaluations
  - D* to test alternative implementations of norms (ie. forbid neg / oblig pos; deontic vs. factual)
  - E* to test norms inspired to represent concrete paradoxes mentioned in paper (these include also the weird ones)


