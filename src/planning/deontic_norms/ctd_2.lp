% Chisholm, deontic
% Checks:
% occupiedTraverserTile
% turnedOnTraverserTile
% notReachedGoal

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

conflict(X,Y) :- conflict(Y,X).
superior(X,Y) :- inferior(Y,X), X != Y.
inferior(X,Y) :- superior(Y,X), X != Y.

conflict(occupyTraverserTile, turnedOnTraverserTile).
superior(forbidden(turnedOnTraverserTile),forbidden(occupyTraverserTile)).

% The agent must not be on the same tile as the traverser
forbidden(occupyTraverserTile).
occupyTraverserTile :- currentState(X), currentStateOfTraverser(Y), X=Y.

% If the agent and traverser occupy the same tile, then the agent must move straight
forbidden(turnedOnTraverserTile).
turnedOnTraverserTile :- occupyTraverserTile, act(move(X)), lastPerformedAction(move(Y)), X != Y.
% TODO: check if lastPerformedAction is the same as 'act(move(..)) !

currentViolation(forbidden(X)) :- forbidden(X), X.

% The agent must reach the goal tile
obligatory(reachedGoal).
#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.

% :~ currentViolation(forbidden(occupyTraverserTile)). [1@1]
% :~ currentViolation(forbidden(turnedOnTraverserTile)). [1@3]
% :- currentViolation(forbidden(X)), superior(forbidden(X),forbidden(Y)), forbidden(Y), not currentViolation(forbidden(Y)), conflict(X,Y), X != Y.
