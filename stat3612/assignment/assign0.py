import numpy as np

# Q1
## a
q1a = np.full((2,5), 5)

## b
q1b = np.arange(1, 26).reshape(5, 5)

## c
q1c = q1b[0, :3]

## d
q1d = q1b[:2, 0::2]

## e
q1e = np.diag(q1b)

## f
q1f = np.arange(1, 26).reshape(5, 1, 5)

## g
q1h = np.squeeze(q1f)

## h
h1 = [[1, 3, 5],
      [6, 8, 10],
      [11, 13, 15]]
h2 = [[2, 6, 10],
      [12, 16, 20],
      [22, 26, 30]]
q1h1 = np.vstack((h1, h2))
q1h2 = np.hstack((h1, h2))

# Q2
## a
a1 = [[1, 2, 4],
      [3, 4, 2]]
a2 = [[4],
      [2],
      [1]]
q2a = np.dot(a1, a2)

## b
b1 = [[1],
      [2],
      [4]]
b2 = [[3],
      [4],
      [2]]
#q2b = np.dot(b1, b2)

## c
c1 = [[1, 2, 4],
      [3, 4, 2]]
q2c = np.argmax(c1, axis=0)

## d
d1 = np.array([[6, -1, 10],
               [2, 12, 1],
               [11, 3, -5]])
q2d = d1[(d1 >= 0) & (d1 <= 10)]

# Q3
matrix = np.array([[1.2, 2.6, 4.9],
                   [3.5, 4.7, 2.6],
                   [1.9, 2.5, 4.4],
                   [3.3, 4.5, 2.9],
                   [1.7, 2.6, 4.4]])
## a

print(q2d)