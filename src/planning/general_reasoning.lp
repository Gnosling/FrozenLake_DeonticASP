#program initial.
rewards(X) :- nextReward(X).
violations(X) :- nextViolations(X).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

rewards(X+S) :- 'rewards(X), S = #sum {Y : nextReward(Y)}.
nextViolations(X+Y) :- X = #count {U : currentViolation(forbidden(U))}, Y = #count {W : currentViolation(obligatory(W))}.
violations(X+Y) :- 'violations(X), nextViolations(Y).

#program final.
% was removed to consider short planning_horizon
% :- not terminalStateReached.

#show rewards/1.
#show currentViolation/1.
#show violations/1.
#show nextViolations/1.
