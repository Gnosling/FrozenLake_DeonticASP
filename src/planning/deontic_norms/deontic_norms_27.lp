% forbidden(movedAwayFromGoal). with forbidden(failedToReachGoal).

% Norms:
% notReachedGoal
% movedAwayFromGoal

#program always.

% The agent must not move away from the goal
forbidden(movedAwayFromGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
currentViolation(forbidden(movedAwayFromGoal)) :- forbidden(movedAwayFromGoal), movedAwayFromGoal.
level(movedAwayFromGoal,1).


% The agent must not fail to reach the goal tile
forbidden(failedToReachGoal).
failedToReachGoal :- not goalStateReached.
level(failedToReachGoal,4).

#program final.
currentViolation(forbidden(failedToReachGoal)) :- failedToReachGoal.