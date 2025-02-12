% Norms:
% missedPresents
% notReachedGoal

#program always.

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,2).

% The agent must reach the goal tile
obligatory(reachedGoal).
level(reachedGoal,4).

#program final.
currentViolation(obligatory(reachedGoal)) :- not &tel {<? goalStateReached}.
missedPresents :- present(S), not presentTaken(S).
currentViolation(forbidden(missedPresents)) :- forbidden(missedPresents), missedPresents.