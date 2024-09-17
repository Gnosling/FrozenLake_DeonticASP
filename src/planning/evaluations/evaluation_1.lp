% only rewards

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


gain(G) :- rewards(R), G = 1 * R.
eval(E) :- gain(G), E = G.

#program final.
#maximize {E@1 : eval(E)}.

#show eval/1.


