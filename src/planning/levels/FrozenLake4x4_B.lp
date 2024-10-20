#program always.
#const tiles = 16.
#const width = 4.
#const height = 4.

tile(0..tiles-1).

frozen(0..3).
frozen(4). hole(5). frozen(6). hole(7).
frozen(8..10). hole(11).
frozen(12..15).

goal(15).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.