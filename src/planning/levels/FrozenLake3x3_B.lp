#program always.
#const tiles = 9.
#const width = 3.
#const height = 3.

tile(0..tiles-1).

frozen(0). frozen(1). cracked(2).
frozen(3). hole(4). cracked(5).
cracked(6). cracked(7). frozen(8).

goal(8).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.