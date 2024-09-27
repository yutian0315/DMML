import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D dataset
np.random.seed(0)
class1 = np.random.randn(10, 3) + np.array([1, 1, 1])
class2 = np.random.randn(10, 3) + np.array([5, 5, 5])

# Combine the datasets
X = np.vstack((class1, class2))
y = np.array([0]*10 + [1]*10)

# Plotting the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for class1 and class2
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], color='b', label='Class 1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], color='r', label='Class 2')

# Create a meshgrid for the hyperplane
xx, yy = np.meshgrid(np.linspace(0, 6, 10), np.linspace(0, 6, 10))
zz = (-1/1) * (1 * xx + 1 * yy - 6) # Example hyperplane equation

# Plot the hyperplane
ax.plot_surface(xx, yy, zz, color='g', alpha=0.5)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Hyperplane')
ax.legend()

# Show plot
plt.show()
