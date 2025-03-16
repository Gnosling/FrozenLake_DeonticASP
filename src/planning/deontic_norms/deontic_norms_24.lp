% Ross: A implies A or B, thus O(A) will be O(A or B)

% Norms:
% notReachedGoal

#program always.
obligatory(reachedGoal).
level(reachedGoal,3).

reachedGoal; accomplishedSomethingElse :- reachedGoal.

#program final.
currentViolation(obligatory(reachedGoal)) :- not reachedGoal, not accomplishedSomethingElse.