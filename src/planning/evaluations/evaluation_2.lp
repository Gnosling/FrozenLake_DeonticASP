% scaling rewards with violations

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


penaltyPerLevel(X,L) :- X = #sum { V * U : violationsOfLevel(V,U), U=L}, violationsOfLevel(_,L).
penalty(P) :- P = #sum { X, L : penaltyPerLevel(X,L) }.
benefits(G) :- rewards(R), G = 1 * R.

eval(E) :- penalty(P), benefits(G), E = G - P.
#show penalty/1.
#show penaltyPerLevel/2.

#program final.
#maximize {E@1 : eval(E)}.
