obligatory(reachedGoal).
reachedGoal; failedAtCrossing :- reachedGoal.

#program final.
reachedGoal.
level(reachedGoal,3).
currentViolation(obligatory(reachedGoal)) :- failedAtCrossing.

% :~ currentViolation(obligatory(reachedGoal)). [1@1]