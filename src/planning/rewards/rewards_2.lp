#program initial.
rewards(X) :- nextReward(X).

#program always.
nextReward(0) :- not goalStateReached.
nextReward(100) :- goalStateReached.
nextReward(-100) :- failedStateReached.
nextReward(30) :- takePresent(S), currentState(S).

rewards(X+S) :- 'rewards(X), S = #sum {Y : nextReward(Y)}.