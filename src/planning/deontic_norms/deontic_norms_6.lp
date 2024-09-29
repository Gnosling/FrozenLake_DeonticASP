% moving to goal

% Checks:
% notReachedGoal
% movedAwayFromGoal

#program always.

% The agent must not move away from the goal
forbidden(movedAwayFromGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
currentViolation(forbidden(movedAwayFromGoal)) :- forbidden(movedAwayFromGoal), movedAwayFromGoal.
level(movedAwayFromGoal,2).


% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,2).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.