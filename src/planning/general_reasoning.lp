#program initial.
reward(0).
violations(0).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).

#program dynamic.
reward(X+Y) :- 'reward(X), nextReward(Y).
nextViolations(X+Y) :- X = #count {U : currentViolation(forbidden(U))}, Y = #count {W : currentViolation(obligatory(W))}.
violations(X+Y) :- 'violations(X), nextViolations(Y).

#program final.
:- not terminalStateReached.
#minimize {V@1 : violations(V)}.
#maximize {R@2 : reward(R)}.
% Note: @p_i is priority, higher values have higher prio

#show reward/1.
#show currentViolation/1.
#show violations/1.
