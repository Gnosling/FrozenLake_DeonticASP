% safe over traverser

% Norms:
% notReachedGoal
% occupiedTraverserTile
% turnedOnTraverserTile
% leftSafeArea
% didNotReturnToSafeArea

#program always.

% The agent must not be on the same tile as the traverser
forbidden(occupiedTraverserTile).
occupiedTraverserTile :- currentState(X), currentStateOfTraverser(Y), X=Y.
currentViolation(forbidden(occupiedTraverserTile)) :- forbidden(occupiedTraverserTile), occupiedTraverserTile.
level(occupiedTraverserTile,1).

% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile).
turnedOnTraverserTile :- occupiedTraverserTile, act(move(X)), lastPerformedAction(move(Y)), X != Y.
currentViolation(forbidden(turnedOnTraverserTile)) :- forbidden(turnedOnTraverserTile), turnedOnTraverserTile.
level(turnedOnTraverserTile,2).

% The agent must not leave the safe area
forbidden(leftSafeArea).
leftSafeArea :- currentState(X), not safe(Y), tile(Y), X=Y, 'currentState(U), safe(V), U=V.
currentViolation(forbidden(leftSafeArea)) :- forbidden(leftSafeArea), leftSafeArea.
level(leftSafeArea,2).

% If the agent left the safe area, it must return to a safe tile
forbidden(didNotReturnToSafeArea).
didNotReturnToSafeArea :- currentState(X), not safe(Y), tile(Y), X=Y, 'currentState(U), not safe(V), tile(V), U=V.
currentViolation(forbidden(didNotReturnToSafeArea)) :- forbidden(didNotReturnToSafeArea), didNotReturnToSafeArea.
level(didNotReturnToSafeArea,3).

% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.