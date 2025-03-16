% Norms:
% notReachedGoal
% missedPresents
% movedAwayFromGoal
% stolePresent

#program always.

% The agent must not take any present
forbidden(stolePresent).
stolePresent :- takePresent(S).
currentViolation(forbidden(stolePresent)) :- forbidden(stolePresent), stolePresent.
level(stolePresent,3).

% The agent must not move away from the goal
forbidden(movedAwayFromGoal).
movedAwayFromGoal :- Y > X, 'distanceFromCurrentStateToGoal(X), distanceFromCurrentStateToGoal(Y).
currentViolation(forbidden(movedAwayFromGoal)) :- forbidden(movedAwayFromGoal), movedAwayFromGoal.
level(movedAwayFromGoal,1).

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,2).

% The agent must reach the goal tile
obligatory(reachedGoal).
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.
currentViolation(forbidden(missedPresents(S))) :- forbidden(missedPresents), presentMissed(S).
level(missedPresents(S), L) :- level(missedPresents,L), presentMissed(S).