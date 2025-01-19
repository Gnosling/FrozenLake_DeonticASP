#program always.
#const tiles = 64.
#const width = 8.
#const height = 8.

tile(0..tiles-1).

frozen(0..7).
frozen(8..15).
frozen(16..18). hole(19). frozen(20..21). cracked(22..23).
frozen(24..28). hole(29). cracked(30..31).
frozen(32..34). hole(35). frozen(36..37). cracked(38..39).
frozen(40). hole(41..42). frozen(43..47).
frozen(48). hole(49). frozen(50..51). hole(52). frozen(53..55).
frozen(56..58). hole(59). frozen(60..63).


goal(63).

leftEdge(X) :- tile(X), X \ width = 0.
rightEdge(X) :- tile(X), X \ width = width-1.
upperEdge(X) :- tile(X), X < width.
lowerEdge(X) :- tile(X), X >= tiles-width.