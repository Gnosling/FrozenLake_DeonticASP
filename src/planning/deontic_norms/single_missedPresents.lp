% Norms:
% missedPresents

#program always.

% The agent must take all available presents
forbidden(missedPresents).
level(missedPresents,3).

#program final.
missedPresents :- present(S), not presentTaken(S).
currentViolation(forbidden(missedPresents)) :- forbidden(missedPresents), missedPresents.
