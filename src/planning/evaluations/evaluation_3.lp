% using weak constraints

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


eval(E) :- rewards(R), E = 1 * R.
:~ currentViolation(obligatory(X)), level(X,L). [10@L]
:~ currentViolation(forbidden(X)), level(X,L). [10@L]


#program final.
#maximize {E@1 : eval(E)}.

