import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_moons

# Step 1: Generate a Non-Gaussian Dataset (e.g., Two Moons)
X, Y = make_moons(n_samples=1000, noise=0.01, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Use Kernel Density Estimation (KDE) for Each Class
kde_0 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_train[Y_train == 0])
kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_train[Y_train == 1])

# Step 3: Estimate the Class Probabilities P(X | Y)
log_p_x_given_y0 = kde_0.score_samples(X_test)
log_p_x_given_y1 = kde_1.score_samples(X_test)

# Convert log-likelihoods to probabilities
p_x_given_y0 = np.exp(log_p_x_given_y0)
p_x_given_y1 = np.exp(log_p_x_given_y1)

# Step 4: Compute the Prior Probabilities P(Y)
p_y0 = np.mean(Y_train == 0)
p_y1 = np.mean(Y_train == 1)

# Step 5: Compute Posterior Probabilities P(Y | X) using Bayes' Theorem
p_y0_given_x = (p_x_given_y0 * p_y0) / (p_x_given_y0 * p_y0 + p_x_given_y1 * p_y1)
p_y1_given_x = (p_x_given_y1 * p_y1) / (p_x_given_y0 * p_y0 + p_x_given_y1 * p_y1)

# Step 6: Compute the Bayesian Error Rate
bayes_error = np.mean(np.minimum(p_y0_given_x, p_y1_given_x))

# Step 7: Print the Estimated Bayesian Error
print(f"Estimated Bayesian Error Rate: {bayes_error:.4f}")

# Optional: Plot the dataset
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='coolwarm', alpha=0.6)
plt.title("Test Data Distribution (Color Represents True Labels)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
