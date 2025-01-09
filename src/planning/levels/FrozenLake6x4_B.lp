#program always.
#const tiles = 24.
#const width = 6.
#const height = 4.

tile(0..tiles-1).

frozen(0..3). cracked(4). frozen(5).
frozen(6). hole(7..9). cracked(10). hole(11).
frozen(12). hole(13). frozen(14..16). hole(17).
frozen(18..20). hole(21). frozen(22..23).


goal(23).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.