% Checks:
% occupiedTraverserTile
% turnedOnTraverserTile
% notReachedGoal

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


% The agent must not be on the same tile as the traverser
forbidden(occupyTraverserTile).
currentViolation(forbidden(occupiedTraverserTile)) :- forbidden(occupiedTraverserTile), currentState(X), currentStateOfTraverser(Y), X=Y.


% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile).
currentViolation(forbidden(turnedOnTraverserTile)) :- act(move(X)), 'act(move(Y)), X != Y, currentState(W), currentStateOfTraverser(U), W=U.
currentViolation(forbidden(turnedOnTraverserTile)) :- act(move(X)), lastPerformedAction(move(Y)), X != Y, currentState(W), currentStateOfTraverser(U), W=U.


% The agent must reach the goal tile
obligatory(reachedGoal).
#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.