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
    - shaped_rewards: f(goal) = 1 
  - M: occupiedTraverserTile
    - shaped_rewards: f(free_tile) = 1
  - H: turnedOnTraverserTile (CTD of above)
    - shaped_rewards: not possible in potential-based, f'(not_turned_on_traverser) = 1
  - M: stolePresent (took at least one) (conflicting below) (Agent always picks up first)
    - shaped_rewards: f(state) = number_of_left_presents
  - M: missedPresents (did not take all available) (conflicting above)
    - shaped_rewards: f(state) = -number_of_left_presents
  - L: movedAwayFromGoal (the distance to the goal tile has increased)
    - shaped_rewards: f(s) = -distance_to_goal(s)/0.8
  - L: leftSafeArea ('safe' iff not near a hole, only triggers on exit)
    - shaped_rewards: f(safe) = 1
  - M: didNotReturnToSafeArea (CTD of above)
    - shaped_rewards: not possible in potential-based, f'(returned_to_safe_area ) = 1


- Evaluation:
  - sum of rewards
  - scaled eval of rewards and violations -> favors rewards
  - level-grouping of violations
  - rewards and violations of level (because the weak constraints work differently in Telingo this simulates that) -> favors violations
  - same as before plus a 'hard' conflict resolution per level (this might also worsen higher levels, but interesting to see what happens if all have same level)

---------------

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
  - compared to others high computational effort, but most flexible and best monitoring 
- --> reward_shaping (paper-one):
  - as long as function is potential-based, there is no drop in optimal policy and in fact learning can be speed up by this
  - What is potential-based for FrozenLake?
    - with uncertainty, only positive linear transformations, but the state-function diff handles that already
    - reward of transitions is the difference of function output of both states
    - this guarantees policy invariance of original and shaped
    - in each step: old_rewards + enforced_rewards
    - instead of actions, states are evaluated: discounted_F(successor) - F(prev_state)
    - these plus-rewards can be scaled by the level of the norm and the sum can be downsized
    - However CTDs, can not be expressed by this! -> have two options: optimal_rs and full_rs which is no longer optimal

<br/>
<br/>

###### movedAwayFromGoal vs didNotMoveTowardsGoal
maybe we can experiment with both of these norms? in planning the first one allows stalling as well
also have separate files for each norm to make copying into sets easier


###### Some other notes
- State-Representation:
- --> this is tuple (agent, traverser, (presents))
- --> state space should be fine since it's: Tiles×Tiles×(Possible Configurations of Presents), at worst T^2 x 2^T x actions, but there aren't that many presents there
- --> use at most 3 presents, maybe less then it's fine
- If the planning horizon is too low (or also the enforcing one), planning might prefer to not move away from start -> min-steps count also start tile
- If highest norm is preventing reaching the goal then the planning might stall (If reachedGoal is strictly best, then this works)
- Discount < 1 distords rewards shaping a bit, since equal violations after discounting still give a benefit
- Norms with level 1 are on the same group as internal rewards


<br/>
<br/>

---------------
### Paradoxes:
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
### Experiments:
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


