#program final.
:- not terminalStateReached.
#minimize {V@1 : violations(V)}.
#maximize {R@2 : reward(R)}.
% Note: @p_i is priority, higher values have higher prio