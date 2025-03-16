% Norms:
% missedPresents

#program always.
% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,3).

#program final.
currentViolation(forbidden(missedPresents(S))) :- forbidden(missedPresents), presentMissed(S).
level(missedPresents(S), L) :- level(missedPresents,L), presentMissed(S).