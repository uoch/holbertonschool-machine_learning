#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = matrix[1]
a=5
c = 6
x = [the_middle[i:i+2] for i in range(0, len(the_middle), 2)]
