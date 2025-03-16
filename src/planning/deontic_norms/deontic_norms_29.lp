% obligatory(movedTowardsGoal). with obligatory(reachedGoal).

% Norms:
% notReachedGoal
% movedAwayFromGoal

#program always.

% The agent must not move towards the goal
obligatory(movedTowardsGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
movedTowardsGoal :- not movedAwayFromGoal.
currentViolation(obligatory(movedTowardsGoal)) :- obligatory(movedTowardsGoal), not movedTowardsGoal.
level(movedAwayFromGoal,1).


% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.