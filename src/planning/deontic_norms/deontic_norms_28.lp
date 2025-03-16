% obligatory(movedTowardsGoal). with forbidden(failedToReachGoal).

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


% The agent must not fail to reach the goal tile
forbidden(failedToReachGoal).
failedToReachGoal :- not goalStateReached.
level(failedToReachGoal,4).

#program final.
currentViolation(forbidden(failedToReachGoal)) :- failedToReachGoal.