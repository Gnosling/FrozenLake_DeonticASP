#program always.
forbidden(occupyTraverserTile).
currentViolation(forbidden(occupyTraverserTile)) :- forbidden(occupyTraverserTile), currentState(X), currentStateOfTraverser(Y), X=Y.
forbidden(turnOnTraverserTile).
currentViolation(forbidden(turnOnTraverserTile)) :- act(move(X)), 'act(move(Y)), X != Y, currentState(W), currentStateOfTraverser(U), W=U.
currentViolation(forbidden(turnOnTraverserTile)) :- act(move(X)), lastPerformedAction(move(Y)), X != Y, currentState(W), currentStateOfTraverser(U), W=U.


obligatory(reachedGoal).

#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.