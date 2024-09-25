% using weak constraints

#program always.
:- forbidden(X), obligatory(X).
forbidden(X) :- 'forbidden(X).
obligatory(X) :- 'obligatory(X).


eval(E) :- rewards(R), E = 1 * R.
%:~ currentViolation(obligatory(X)), level(X,L). [6@L]
%:~ currentViolation(forbidden(X)), level(X,L). [1@L]

% TODO: analyse why what happens with this if multiple violations of same type happen -> either the level or the weight must be different?
% TODO: also only the final values count, if currentViolation happened before it vanishes afterwards
% TODO: idea 1 count violations of same level and minimize those at the end with the grouping level -> Done works apparently
% TODO: idea 2 use ordering to define level of each norm, full ordering --> the conflict hard restriction is thus only an extra, but this is bad since opt is only done at the last step
%  State 2:
%  act(move(down))
 % currentState(8)
  %currentStateOfTraverser(14)
%  currentViolation(forbidden(stolePresent)) currentViolation(obligatory(reachedGoal))
 % nextViolations(2)
  %rewards(27)
%  takePresent(8)
 % violations(2)
% Optimization: 10 -27


#program final.
#maximize {E@1 : eval(E)}.
#minimize {V@L : violationsOfLevel(V,L)}.

