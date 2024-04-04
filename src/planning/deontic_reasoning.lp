#program always.
forbidden(currentState(1)).
currentViolation(forbidden(currentState(1))) :- forbidden(currentState(1)), currentState(1).

obligatory(reachedTopRight).
obligatory(reachedBottomLeft).

% TODO: parameterize these states

#program final.
currentViolation(obligatory(reachedTopRight)) :- not &tel {<? currentState(2)}.
currentViolation(obligatory(reachedBottomLeft)) :- not &tel {<? currentState(6)}.