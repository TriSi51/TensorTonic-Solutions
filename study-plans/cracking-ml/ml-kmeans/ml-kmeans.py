import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    X = np.asarray(X, dtype = np.float64)
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(len(X), size = k, replace = False)].copy()
    print("centroids:",centroids)
    for _ in range(max_iters):
        distances = np.sum((X[:,None,:] - centroids[None,:,:]) ** 2, axis = 2)
        print("distances:",distances)
        label = np.argmin(distances, axis = 1)
        print("label:",label)
        updated = centroids.copy()
        
        for cluster in range(k):
            X_cluster = X[label == cluster]
            print("X_cluster:",X_cluster)
            assert updated.shape[1] == X_cluster.shape[1]
            if X_cluster.shape[0] > 0:
                updated[cluster] = np.mean(X_cluster, axis = 0)

        if np.allclose(updated, centroids):
            centroids = updated
            break
        centroids = updated
    distances = np.sum((X[:,None,:] - centroids[None,:,:]) ** 2, axis = 2)
    label = np.argmin(distances, axis = 1)
    
    return (label.tolist(), [[round(float(centroids[i,j]),4) for j in range(centroids.shape[1])] for i in range(centroids.shape[0])])  
            
            

        
        
    
    
    