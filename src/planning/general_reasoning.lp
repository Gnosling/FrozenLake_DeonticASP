#program initial.
reward(0).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

#program dynamic.
reward(X+Y) :- 'reward(X), nextReward(Y).

#program final.
:- not terminalStateReached.
#maximize { R : reward(R) }.

% #maximize [ hotel(X) : star(X,Y) = Y @ 1 ].           lowest prio
% #minimize [ hotel(X) : cost(X,Y) : star(X,Z) = Y/Z @ 2 ].
% #minimize { noisy @ 3 }.                              highest prio

#show reward/1.

