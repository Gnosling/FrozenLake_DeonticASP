#program always.
#const tiles = 24.
#const width = 6.
#const height = 4.

tile(0..tiles-1).

frozen(0). hole(1..2). frozen(3..5).
frozen(6). hole(7). frozen(8..11).
frozen(12..15). hole(16). frozen(17).
frozen(18..20). hole(21..22). frozen(23).


goal(23).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.