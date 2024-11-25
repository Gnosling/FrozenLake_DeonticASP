#show violationsOfLevel/2.
#show currentViolation/1.
#show currentState/1.
#show currentStateOfTraverser/1.
#show act/1.
#show takePresent/1.
#show presentTaken/1.
#show distanceFromCurrentStateToGoal/1.
#show stepCounter/1.
#show givenAction/2.
#show sumOfViolations/1.



#program initial.
violationsOfLevel(X,L) :- nextViolationsOfLevel(X,L).
stepCounter(0).


#program always.
act(move(X)) :- givenAction(X,C), stepCounter(C).
stepCounter(C+1) :- 'stepCounter(C).

nextViolationsOfLevel(X+Y,L) :- X = #count {U : currentViolation(forbidden(U)), level(U,L)}, Y = #count {W : currentViolation(obligatory(W)), level(W,L)}, level(_,L).
violationsOfLevel(X+Y,L) :- 'violationsOfLevel(X,L), nextViolationsOfLevel(Y,L).

sumOfViolations(S) :- S = #sum { X : violationsOfLevel(X,_)}.

% assume traverser and presents don't move
currentStateOfTraverser(X) :- 'currentStateOfTraverser(X).
present(S) :- 'present(S).

% define transitions
currentState(L-1) :- 'act(move(left)), 'currentState(L), not leftEdge(L).
currentState(L) :- 'act(move(left)), 'currentState(L), leftEdge(L).
currentState(L+1) :- 'act(move(right)), 'currentState(L), not rightEdge(L).
currentState(L) :- 'act(move(right)), 'currentState(L), rightEdge(L).
currentState(L-width):- 'act(move(up)), 'currentState(L), not upperEdge(L).
currentState(L):- 'act(move(up)), 'currentState(L), upperEdge(L).
currentState(L+width) :- 'act(move(down)), 'currentState(L), not lowerEdge(L).
currentState(L) :- 'act(move(down)), 'currentState(L), lowerEdge(L).
:- currentState(X), currentState(Y), X != Y.


% define termination
goalStateReached :- currentState(S), goal(S).
failedStateReached :- currentState(S), hole(S).
failedStateReached :- currentState(S), currentStateOfTraverser(S), cracked(S).
terminalStateReached :- goalStateReached.
terminalStateReached :- failedStateReached.
:- failedStateReached.

% handle presents (agent always picks up first)
takePresent(S) :- currentState(S), present(S), not presentTaken(S).
presentTaken(S) :- 'takePresent(S).
presentTaken(S) :- 'presentTaken(S).
presentTaken(S) :- currentStateOfTraverser(S), present(S), not presentTaken(S), not currentState(S).


% define safe tiles as tiles not next to a hole
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S-1), not hole(S+width), not hole(S-width), not leftEdge(S), not lowerEdge(S), not rightEdge(S), not upperEdge(S).
% Special cases for edges:
safe(S) :- tile(S), not hole(S), not hole(S-1), not hole(S+width), not hole(S-width), rightEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S+width), not hole(S-width), leftEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S-1), not hole(S+width), upperEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S-1), not hole(S-width), lowerEdge(S).
% Special cases for corners:
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S+width), upperEdge(S), leftEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S-1), not hole(S+width), upperEdge(S), rightEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S+1), not hole(S-width), lowerEdge(S), leftEdge(S).
safe(S) :- tile(S), not hole(S), not hole(S-1), not hole(S-width), lowerEdge(S), rightEdge(S).


% define distance to goal
distanceFromCurrentStateToGoal(DIS) :- DIS = X + Y, X = |(S\width) - (G\width)|, Y = |(S/height) - (G/height)|, currentState(S), goal(G).


#program dynamic.
lastPerformedAction(move(X)) :- 'act(move(X)). % this is needed since the first last action comes from the environment

#program final.
:- sumOfViolations(S), S > 0.