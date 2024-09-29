% scaling rewards with violations

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


% TODO: update file with maximize and use single violations as penalty

penalty(P) :- violations(V), P = 50 * V.
benefits(G) :- rewards(R), G = 1 * R.

eval(E) :- penalty(P), benefits(G), E = G - P.

#program final.
#maximize {E@1 : eval(E)}.