#program initial.
rewards(X) :- nextReward(X).
violationsOfLevel(X,L) :- nextViolationsOfLevel(X,L).

#program always.
{act(move(X)) : executable(move(X))} = 1 :- not terminalStateReached.

rewards(X+S) :- 'rewards(X), S = #sum {Y : nextReward(Y)}.
nextViolationsOfLevel(X+Y,L) :- X = #count {U : currentViolation(forbidden(U)), level(U,L)}, Y = #count {W : currentViolation(obligatory(W)), level(W,L)}, level(_,L).
violationsOfLevel(X+Y,L) :- 'violationsOfLevel(X,L), nextViolationsOfLevel(Y,L).

#program final.
% was removed to consider short planning_horizon
% :- not terminalStateReached.

#show rewards/1.
#show currentViolation/1.
#show violationsOfLevel/2.
#show nextViolationsOfLevel/2.
