# Discovered Seven-Multiplication Algorithm over GF(2)

All additions below are XOR operations. Matrix entries use row/column names
`a11, a12, a21, a22` and `b11, b12, b21, b22`.

## Seven scalar multiplications

1. `p1 = a22 * b21`
2. `p2 = a21 * b11`
3. `p3 = (a12 + a22) * (b11 + b12 + b21 + b22)`
4. `p4 = a11 * (b12 + b22)`
5. `p5 = (a11 + a12) * b22`
6. `p6 = (a11 + a12 + a22) * (b11 + b12 + b22)`
7. `p7 = (a11 + a12 + a21 + a22) * (b11 + b12)`

## Output reconstruction

- `c11 = p1 + p3 + p4 + p6`
- `c12 = p4 + p5`
- `c21 = p1 + p2`
- `c22 = p2 + p5 + p6 + p7`

This algorithm was verified by exact tensor reconstruction and by testing all
256 pairs of binary 2 x 2 input matrices.
