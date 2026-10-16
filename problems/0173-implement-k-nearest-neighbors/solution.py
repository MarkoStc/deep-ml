import numpy as np
import heapq

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    query_point = np.array(query_point)
    h = []
    for i, p in enumerate(points):
        p = np.array(p)
        d = np.linalg.norm(p-query_point)
        heapq.heappush(h, (d, i))
    ans = []
    for _ in range(k):
        d, i = heapq.heappop(h)
        ans.append(points[i])
    return ans









    