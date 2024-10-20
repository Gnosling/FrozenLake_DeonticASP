% presents

% Norms:
% notReachedGoal
% stolePresent
% missedPresents

#program always.

% The agent must not take any present
forbidden(stolePresent).
stolePresent :- takePresent(S).
currentViolation(forbidden(stolePresent)) :- forbidden(stolePresent), stolePresent.
level(stolePresent,2).

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,2).

% The agent must reach the goal tile
obligatory(reachedGoal).
reachedGoal :- goalStateReached.
level(reachedGoal,2).

#program final.
missedPresents :- present(S), not presentTaken(S).
currentViolation(forbidden(missedPresents)) :- forbidden(missedPresents), missedPresents.

currentViolation(obligatory(reachedGoal)) :- not reachedGoal.