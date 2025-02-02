#program always.
#const tiles = 28.
#const width = 7.
#const height = 4.

tile(0..tiles-1).

frozen(0..2). cracked(3). frozen(4..6).
frozen(7). hole(8..12). frozen(13).
frozen(14). hole(15). frozen(16..18). hole(19). frozen(20).
frozen(21..23). hole(24). frozen(25..27).


goal(6).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.