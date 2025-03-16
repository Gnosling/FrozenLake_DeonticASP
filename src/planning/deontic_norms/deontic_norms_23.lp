% Conflict / Satre

% Norms:
% stolePresent
% missedPresents

#program always.

% The agent must not take any present
forbidden(stolePresent).
stolePresent :- takePresent(S).
currentViolation(forbidden(stolePresent)) :- forbidden(stolePresent), stolePresent.
level(stolePresent,3).

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,3).

#program final.
currentViolation(forbidden(missedPresents(S))) :- forbidden(missedPresents), presentMissed(S).
level(missedPresents(S), L) :- level(missedPresents,L), presentMissed(S).