% Chisholm, factual detachment (forbidden(turnedOnTraverserTile) <- occupyTraverserTile)

% Norms:
% occupiedTraverserTile
% turnedOnTraverserTile
% notReachedGoal

#program always.

% The agent must not be on the same tile as the traverser
forbidden(occupyTraverserTile).
occupyTraverserTile :- currentState(X), currentStateOfTraverser(Y), X=Y.
currentViolation(forbidden(occupyTraverserTile)) :- forbidden(occupyTraverserTile), occupyTraverserTile.
level(occupyTraverserTile,2).

% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile) :- occupyTraverserTile.
turnedOnTraverserTile :- occupyTraverserTile, act(move(X)), lastPerformedAction(move(Y)), X != Y.
currentViolation(forbidden(turnedOnTraverserTile)) :- forbidden(turnedOnTraverserTile), turnedOnTraverserTile.
level(turnedOnTraverserTile,3).


% The agent must reach the goal tile
obligatory(reachedGoal).
#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.
level(reachedGoal,2).
