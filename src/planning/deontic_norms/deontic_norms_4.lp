% Norms:
% occupiedTraverserTile
% turnedOnTraverserTile
% missedPresents
% notReachedGoal

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
level(turnedOnTraverserTile,3).

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,2).

% The agent must reach the goal tile
obligatory(reachedGoal).
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.
missedPresents :- present(S), not presentTaken(S).
currentViolation(forbidden(missedPresents)) :- forbidden(missedPresents), missedPresents.
