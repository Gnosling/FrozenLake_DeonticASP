% Norms:
% notReachedGoal
% stolePresent

#program always.

% The agent must not take any present
forbidden(stolePresent).
stolePresent :- takePresent(S).
currentViolation(forbidden(stolePresent)) :- forbidden(stolePresent), stolePresent.
level(stolePresent,3).


% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal.