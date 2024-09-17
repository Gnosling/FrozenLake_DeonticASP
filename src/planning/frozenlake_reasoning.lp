#show currentState/1.
#show currentStateOfTraverser/1.
#show act/1.

#program always.

% define possible actions
executable(move(left)) :- currentState(L), frozen(L-1), not leftEdge(L), not terminalStateReached.
executable(move(down)) :- currentState(L), frozen(L+width), not lowerEdge(L), not terminalStateReached.
executable(move(right)) :- currentState(L), frozen(L+1), not rightEdge(L), not terminalStateReached.
executable(move(up)) :- currentState(L), frozen(L-width), not upperEdge(L), not terminalStateReached.

executable(move(left)) :- currentState(L), cracked(L-1), not leftEdge(L), not terminalStateReached.
executable(move(down)) :- currentState(L), cracked(L+width), not lowerEdge(L), not terminalStateReached.
executable(move(right)) :- currentState(L), cracked(L+1), not rightEdge(L), not terminalStateReached.
executable(move(up)) :- currentState(L), cracked(L-width), not upperEdge(L), not terminalStateReached.

% assume traverser stays
currentStateOfTraverser(X) :- 'currentStateOfTraverser(X).

% define transitions
% currentState(L):- currentState(L), not act(move(_)).
currentState(L-1) :- 'act(move(left)), 'currentState(L).
currentState(L+1) :- 'act(move(right)), 'currentState(L).
currentState(L-width):- 'act(move(up)), 'currentState(L).
currentState(L+width) :- 'act(move(down)), 'currentState(L).
:- currentState(X), currentState(Y), X != Y.


% define termination
goalStateReached :- currentState(S), goal(S).
failedStateReached :- currentState(S), hole(S).
failedStateReached :- currentState(S), currentStateOfTraverser(S), cracked(S).
terminalStateReached :- goalStateReached.
terminalStateReached :- failedStateReached.


% define rewards
nextReward(-1) :- not goalStateReached.
nextReward(100) :- goalStateReached.
nextReward(-100) :- failedStateReached.
% TODO: have neg reward for failedStateReached? -> yeah we can just set +/-100 for final things