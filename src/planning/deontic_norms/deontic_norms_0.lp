% Full set of norms

% Norms:
% notReachedGoal
% occupiedTraverserTile
% turnedOnTraverserTile
% movedAwayFromGoal
% didNotMoveTowardsGoal
% leftSafeArea
% didNotReturnToSafeArea
% stolePresent
% missedPresents

#program always.

% The agent must not move away from the goal
forbidden(movedAwayFromGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
currentViolation(forbidden(movedAwayFromGoal)) :- forbidden(movedAwayFromGoal), movedAwayFromGoal.
level(movedAwayFromGoal,1).

% The agent must move towards the goal
forbidden(didNotMoveTowardsGoal).
didNotMoveTowardsGoal :- Y >= X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
didNotMoveTowardsGoal :- act(move(left)), currentState(L), leftEdge(L).
didNotMoveTowardsGoal :- act(move(right)), currentState(L), rightEdge(L).
didNotMoveTowardsGoal :- act(move(up)), currentState(L), upperEdge(L).
didNotMoveTowardsGoal :- act(move(down)), currentState(L), lowerEdge(L).
currentViolation(forbidden(didNotMoveTowardsGoal)) :- forbidden(didNotMoveTowardsGoal), didNotMoveTowardsGoal.
level(didNotMoveTowardsGoal,1).

% The agent must not be on the same tile as the traverser
forbidden(occupiedTraverserTile).
occupiedTraverserTile :- currentState(X), currentStateOfTraverser(Y), X=Y.
currentViolation(forbidden(occupiedTraverserTile)) :- forbidden(occupiedTraverserTile), occupiedTraverserTile.
level(occupiedTraverserTile,2).

% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile).
turnedOnTraverserTile :- occupiedTraverserTile, act(move(X)), lastPerformedAction(move(Y)), X != Y.
currentViolation(forbidden(turnedOnTraverserTile)) :- forbidden(turnedOnTraverserTile), turnedOnTraverserTile.
level(turnedOnTraverserTile,3).

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

% The agent must not take any present
forbidden(stolePresent).
stolePresent :- takePresent(S).
currentViolation(forbidden(stolePresent)) :- forbidden(stolePresent), stolePresent.
level(stolePresent,3).

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,3).

% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.
currentViolation(forbidden(missedPresents(S))) :- forbidden(missedPresents), presentMissed(S).
level(missedPresents(S), L) :- level(missedPresents,L), presentMissed(S).
