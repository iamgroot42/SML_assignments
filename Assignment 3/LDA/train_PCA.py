from sklearn.decomposition import PCA


def fitPCA(X):
        pca = PCA()
        pca.fit(X)
        return pca
