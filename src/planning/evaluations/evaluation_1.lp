% only rewards

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


eval(E) :- rewards(R), E = 1 * R.

#program final.
#maximize {E@1 : eval(E)}.

#show eval/1.


