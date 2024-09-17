% Checks:
% notReachedGoal

#program always.
% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,3).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.
