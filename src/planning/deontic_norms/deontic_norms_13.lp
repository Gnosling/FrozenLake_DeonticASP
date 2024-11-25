% safe area

% Norms:
% leftSafeArea
% didNotReturnToSafeArea
% didNotMoveTowardsGoal

#program always.

% The agent must not leave the same area
forbidden(leftSafeArea).
leftSafeArea :- currentState(X), not safe(Y), tile(Y), X=Y, 'currentState(U), safe(V), U=V.
currentViolation(forbidden(leftSafeArea)) :- forbidden(leftSafeArea), leftSafeArea.
level(leftSafeArea,2).

% If the agent left the safe area, it must return to a safe tile
forbidden(didNotReturnToSafeArea).
didNotReturnToSafeArea :- currentState(X), not safe(Y), tile(Y), X=Y, 'currentState(U), not safe(V), tile(V), U=V.
currentViolation(forbidden(didNotReturnToSafeArea)) :- forbidden(didNotReturnToSafeArea), didNotReturnToSafeArea.
level(didNotReturnToSafeArea,3).


% The agent must not move away from the goal
forbidden(didNotMoveTowardsGoal).
didNotMoveTowardsGoal :- Y >= X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
didNotMoveTowardsGoal :- act(move(left)), currentState(L), leftEdge(L).
didNotMoveTowardsGoal :- act(move(right)), currentState(L), rightEdge(L).
didNotMoveTowardsGoal :- act(move(up)), currentState(L), upperEdge(L).
didNotMoveTowardsGoal :- act(move(down)), currentState(L), lowerEdge(L).
currentViolation(forbidden(didNotMoveTowardsGoal)) :- forbidden(didNotMoveTowardsGoal), didNotMoveTowardsGoal.
level(didNotMoveTowardsGoal,2).
