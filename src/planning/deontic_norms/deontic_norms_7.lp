% moving to goal

% Norms:
% notReachedGoal
% occupiedTraverserTile
% turnedOnTraverserTile
% movedAwayFromGoal

#program always.

% The agent must not move away from the goal
forbidden(movedAwayFromGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
currentViolation(forbidden(movedAwayFromGoal)) :- forbidden(movedAwayFromGoal), movedAwayFromGoal.
level(movedAwayFromGoal,1).

% The agent must not be on the same tile as the traverser
forbidden(occupyTraverserTile).
occupyTraverserTile :- currentState(X), currentStateOfTraverser(Y), X=Y.
currentViolation(forbidden(occupyTraverserTile)) :- forbidden(occupyTraverserTile), occupyTraverserTile.
level(occupyTraverserTile,2).

% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile).
turnedOnTraverserTile :- occupyTraverserTile, act(move(X)), lastPerformedAction(move(Y)), X != Y.
currentViolation(forbidden(turnedOnTraverserTile)) :- forbidden(turnedOnTraverserTile), turnedOnTraverserTile.
level(turnedOnTraverserTile,3).


% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,3).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.