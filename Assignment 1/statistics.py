import numpy as np
import matplotlib.pyplot as plt

# Fix seed for reproducability
np.random.seed(42)

def generateData(mean, cov, numPoints):
    data = np.random.multivariate_normal(mean, cov, numPoints)
    return data[:,0], data[:,1]

def plotData(x, y, title):
    plt.scatter(x, y, c="r", alpha=0.5, marker='+', label="Data Point")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

def varyingSigma(mean, cov, numPoints, sigmaIndex, sigmaRange, title):
    correlations = []
    for i in range(sigmaRange[0], sigmaRange[1]):
        newCov = np.copy(cov)
        newCov[sigmaIndex[0]][sigmaIndex[1]] = i
        x, y = generateData(mean, newCov, numPoints)
        correlations.append(np.corrcoef(x, y)[0][1])
    plt.plot(range(sigmaRange[0], sigmaRange[1]) \
    , correlations, c="b", label="Correlation Scores")
    plt.xlabel("Varying Sigma")
    plt.ylabel("Correlation Coefficient")
    plt.legend(loc=2)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    mu = [2, 3]
    cov = [[1, 1.5], [1.5, 30]]
    x, y = generateData(mu, cov, 100)
    plotData(x, y, "Randomly Generated Data")
    plotData(x**2, y**2, "Randomly Generated Data ^ 2")
    # Correlation between x & y
    print("Correlation  coefficient:",np.corrcoef(x, y)[0][1])
    # Plot of correlations for varying sigma
    varyingSigma(mu, cov, 100, [1, 1], [-30, 90], "Vary Sigma for Y")
    # Correlation over squared values
    x **= 2
    y **= 2
    print("Correlation  coefficient for (X^2, Y^2):",np.corrcoef(x, y)[0][1])
    # Generate new X,Y with different mu
    mu_new = [-2, 3]
    cov = [[1, 1.5], [1.5, 30]]
    x_new, y_new = generateData(mu_new, cov, 100)
    plotData(x_new, y_new, "Randomly Generated Data (new mu)")
    plotData(x_new**2, y_new**2, "Randomly Generated Data (new mu) ^ 2")
    print("Correlation coefficient for new (X, Y):", np.corrcoef(x_new, y_new)[0][1])
    x_new **= 2
    y_new **= 2
    print("Correlation  coefficient for new (X^2, Y^2):",\
        np.corrcoef(x_new, y_new)[0][1])
