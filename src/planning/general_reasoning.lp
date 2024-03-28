#program initial.
reward(0).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

#program dynamic.
reward(X+Y) :- 'reward(X), nextReward(Y).

#program final.
:- not terminalStateReached.
#maximize { R : reward(R) }.

#show reward/1.

