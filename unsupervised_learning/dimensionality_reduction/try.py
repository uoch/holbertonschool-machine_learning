import numpy as np

# Example array Y with shape (3, 2)
Y = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Adding a new axis
Y_newaxis = Y[np.newaxis, :, :]
y = Y[:, np.newaxis, :]
diff = Y_newaxis - y
print("Original Y:")
print(Y)
print("Shape of Y:", Y.shape)

print("\nY with new axis:")
print(Y_newaxis)
print("Shape of Y with new axis:", Y_newaxis.shape)

print("\nY with new axis:")
print(y)
print("Shape of Y with new axis:", y.shape)

print("\nDifference between Y with new axis and Y with new axis:")
print(diff)
print("Shape of difference between Y with new axis and Y with new axis:",
      diff.shape)

