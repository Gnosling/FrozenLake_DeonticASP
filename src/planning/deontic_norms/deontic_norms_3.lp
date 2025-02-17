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
currentViolation(forbidden(missedPresents(S))) :- forbidden(missedPresents), presentMissed(S).
level(missedPresents(S), L) :- level(missedPresents,L), presentMissed(S).