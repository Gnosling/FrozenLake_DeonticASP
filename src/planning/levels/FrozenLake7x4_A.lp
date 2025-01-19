#program always.
#const tiles = 28.
#const width = 7.
#const height = 4.

tile(0..tiles-1).

frozen(0..1). hole(2..4). frozen(5..6).
frozen(7..13).
hole(14). frozen(15..19). hole(20).
hole(21..22). frozen(23..25). hole(26..27).


goal(7).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.