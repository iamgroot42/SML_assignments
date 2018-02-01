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
    mean = [2, 3]
    cov = [[1, 1.5], [1.5, 30]]
    x, y = generateData(mean, cov, 100)
    plotData(x, y, "Randomly Generated Data")
    # Correlation between x & y
    print("Correlation coefficient:",np.corrcoef(x, y)[0][1])
    # Plot of correlations for varying sigma
    varyingSigma(mean, cov, 100, [1, 1], [-50,50], "Vary Sigma for Y")
