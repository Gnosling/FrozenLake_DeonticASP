#program initial.
reward(X) :- nextReward(X). % TODO: not sure if to initialize reward with 0 or -1
violations(X) :- nextViolations(X).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

reward(X+Y) :- 'reward(X), nextReward(Y).
nextViolations(X+Y) :- X = #count {U : currentViolation(forbidden(U))}, Y = #count {W : currentViolation(obligatory(W))}.
violations(X+Y) :- 'violations(X), nextViolations(Y).

#show reward/1.
#show currentViolation/1.
#show violations/1.
#show nextViolations/1.
