#program always.
penalty(P) :- violations(V), P = 50 * V.
gain(G) :- rewards(R), G = 1 * R.

eval(E) :- penalty(P), gain(G), F = G - P, E = -1 * F.

#program final.
#minimize {E@1 : eval(E)}.

%#show penalty/1.
%#show gain/1.
#show eval/1.