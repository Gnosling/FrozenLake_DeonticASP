% Checks:
% notReachedGoal

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

% The agent must reach the goal tile
obligatory(reachedGoal).
#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.