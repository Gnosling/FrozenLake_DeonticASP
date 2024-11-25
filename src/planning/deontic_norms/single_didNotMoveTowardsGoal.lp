% Norms:
% didNotMoveTowardsGoal

#program always.

% The agent must not move away from the goal
forbidden(didNotMoveTowardsGoal).
didNotMoveTowardsGoal :- Y >= X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
didNotMoveTowardsGoal :- act(move(left)), currentState(L), leftEdge(L).
didNotMoveTowardsGoal :- act(move(right)), currentState(L), rightEdge(L).
didNotMoveTowardsGoal :- act(move(up)), currentState(L), upperEdge(L).
didNotMoveTowardsGoal :- act(move(down)), currentState(L), lowerEdge(L).
currentViolation(forbidden(didNotMoveTowardsGoal)) :- forbidden(didNotMoveTowardsGoal), didNotMoveTowardsGoal.
level(didNotMoveTowardsGoal,2).
