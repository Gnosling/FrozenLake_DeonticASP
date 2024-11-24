% only violations

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

#program final.
#minimize {V@L : violationsOfLevel(V,L)}.
