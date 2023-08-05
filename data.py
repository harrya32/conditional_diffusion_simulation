import torch
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import numpy as np

def BOD_generate_data(n):
    x = torch.randn(n, 2)
    A = 0.4 + 0.4 * (1 + torch.erf(x[:, :1] / np.sqrt(2)))
    B = 0.01 + 0.15 * (1 + torch.erf(x[:, 1:] / np.sqrt(2)))
    normal = torch.randn(n, 5) * np.sqrt(1e-3)

    y_mean = A * (1 - torch.exp(-B * torch.arange(1, 6)))
    y = y_mean + normal
    return x, y


def find_max_dist(data):
    
    # Find a convex hull in O(N log N)
    points = data.numpy()

    # Returned 420 points in testing
    hull = ConvexHull(points)

    # Extract the points forming the hull
    hullpoints = points[hull.vertices,:]

    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    return np.linalg.norm(hullpoints[bestpair[0]]-hullpoints[bestpair[1]])