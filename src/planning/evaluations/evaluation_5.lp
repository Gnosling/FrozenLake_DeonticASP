% using weak constraints and conflict-restrictions

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

superior(X,Y) :- inferior(Y,X), X != Y.
inferior(X,Y) :- superior(Y,X), X != Y.


benefits(E) :- rewards(R), E = 1 * R.

:- currentViolation(forbidden(X)), superior(X,Y), forbidden(Y), not currentViolation(forbidden(Y)), X != Y.
:- currentViolation(obligatory(X)), superior(X,Y), forbidden(Y), not currentViolation(forbidden(Y)), X != Y.
:- currentViolation(forbidden(X)), superior(X,Y), obligatory(Y), not currentViolation(obligatory(Y)), X != Y.
:- currentViolation(obligatory(X)), superior(X,Y), obligatory(Y), not currentViolation(obligatory(Y)), X != Y.

#program final.
#maximize {E@1 : benefits(E)}.
#minimize {V@L : violationsOfLevel(V,L)}.

