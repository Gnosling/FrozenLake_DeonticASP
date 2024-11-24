% using weak constraints

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

benefits(E) :- rewards(R), E = 1 * R.


#program final.
#maximize {E@1 : benefits(E)}.
#minimize {V@L : violationsOfLevel(V,L)}.

