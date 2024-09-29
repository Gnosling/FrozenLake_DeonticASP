% using weak constraints and conflict-restrictions

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

conflict(X,Y) :- conflict(Y,X).
superior(X,Y) :- inferior(Y,X), X != Y.
inferior(X,Y) :- superior(Y,X), X != Y.

benefits(E) :- rewards(R), E = 1 * R.

:- currentViolation(forbidden(X)), superior(forbidden(X),forbidden(Y)), forbidden(Y), not currentViolation(forbidden(Y)), conflict(X,Y), X != Y.
:- currentViolation(obligatory(X)), superior(obligatory(X),forbidden(Y)), forbidden(Y), not currentViolation(forbidden(Y)), conflict(X,Y), X != Y.
:- currentViolation(forbidden(X)), superior(forbidden(X),obligatory(Y)), obligatory(Y), not currentViolation(obligatory(Y)), conflict(X,Y), X != Y.
:- currentViolation(obligatory(X)), superior(obligatory(X),obligatory(Y)), obligatory(Y), not currentViolation(obligatory(Y)), conflict(X,Y), X != Y.

#program final.
#maximize {E@1 : eval(E)}.
#minimize {V@L : violationsOfLevel(V,L)}.

