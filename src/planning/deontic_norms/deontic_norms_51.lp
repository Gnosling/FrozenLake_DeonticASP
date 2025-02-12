% safe area

% Norms:
% leftSafeArea
% didNotReturnToSafeArea
% notReachedGoal

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

% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.