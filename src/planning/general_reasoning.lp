#program initial.
violationsOfLevel(X,L) :- nextViolationsOfLevel(X,L).

#program always.
{act(move(X)) : executable(move(X)), allowedAction(move(X))} = 1 :- not terminalStateReached.

nextViolationsOfLevel(X+Y,L) :- X = #count {U : currentViolation(forbidden(U)), level(U,L)}, Y = #count {W : currentViolation(obligatory(W)), level(W,L)}, level(_,L).
violationsOfLevel(X+Y,L) :- 'violationsOfLevel(X,L), nextViolationsOfLevel(Y,L).

#program final.

% was removed to consider short planning_horizon
% :- not terminalStateReached.

#show rewards/1.
#show violationsOfLevel/2.
#show currentViolation/1.
