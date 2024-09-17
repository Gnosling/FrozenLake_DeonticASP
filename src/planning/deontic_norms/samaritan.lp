% Checks:
% occupiedTraverserTile
% notTurnedOnTraverserTile
% notReachedGoal

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


occupyTraverserTile :- currentState(S), currentStateOfTraverser(T), S=T.
occupyTraverserTile :- notTurnedOnTraverserTile
notTurnedOnTraverserTile :- currentState(S), currentStateOfTraverser(T), S=T, act(move(X)), 'act(move(Y)), X == Y.

% The agent must not turn on traverser-tile and agent and traverser are on the same tile
obligatory(notTurnedOnTraverserTile)
currentViolation(obligatory(notTurnedOnTraverserTile)) :- obligatory(notTurnedOnTraverserTile), notTurnedOnTraverserTile.

% check that obligatory(occupyTraverserTile) does not hold
forbidden(occupyTraverserTile)
currentViolation(forbidden(occupyTraverserTile)) :- forbidden(occupyTraverserTile), occupyTraverserTile.


% The agent must reach the goal tile
obligatory(reachedGoal).
#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.