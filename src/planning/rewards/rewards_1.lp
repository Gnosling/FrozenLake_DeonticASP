% same as original frozenlake

#program initial.
rewards(X) :- nextReward(X).

#program always.
nextReward(0) :- not goalStateReached.
nextReward(1) :- goalStateReached.

rewards(X+S) :- 'rewards(X), S = #sum {Y : nextReward(Y)}.