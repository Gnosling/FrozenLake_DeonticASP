obligatory(join).
obligatory(denyInvitation).

% instead of invitation use like going that way or that way if hole is in the middle

conflict(X,Y) :- conflict(Y,X).

% incomparable(X,Y) :- incomparable(Y,X).
% incomparable(X,Y) :- not superior(X,Y), not inferior(X,Y).

superior(X,Y) :- inferior(Y,X), X != Y.
inferior(X,Y) :- superior(Y,X), X != Y.

conflict(join, denyInvitation).
superior(obligatory(join),obligatory(denyInvitation)).

currentViolation(obligatory(join)) :- denyInvitation.
currentViolation(obligatory(denyInvitation)) :- join.


:- currentViolation(obligatory(X)), superior(obligatory(X),obligatory(Y)), obligatory(Y), not currentViolation(obligatory(Y)), conflict(X,Y), X != Y.


{join; denyInvitation} = 1.


