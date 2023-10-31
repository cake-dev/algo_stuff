import numpy as np

class kruskalClass:
    
    def __init__(self):
        pass
    
    def findMinimumSpanningTree(self, A):
        """
        Finds the minimum spanning tree of a graph using Kruskal's algorithm.
        A is the adjacency matrix of the graph.
        """
        N = A.shape[0]
        T = np.zeros((N, N))
        edges = []
        
        # Collect all the edges and their weights from the adjacency matrix
        for i in range(N):
            for j in range(i+1, N):
                if A[i][j] > 0:
                    edges.append((A[i][j], i, j))
                    
        # Sort edges by weight using the implemented merge sort algorithm
        edges = sorted(edges, key=lambda x: x[0])
        
        u = self.makeUnionFind(N)
        
        for edge in edges:
            weight, u, v = edge
            
            # Check if the edge forms a cycle
            if self.find(u, u) != self.find(u, v):
                self.union(u, self.find(u, u), self.find(u, v))
                T[u][v] = weight
                
        return T
    
    def mergesort(self, a):
        """
        Sorts a numpy array in ascending order using the merge sort algorithm.
        """
        if len(a) <= 1:
            return a
        
        mid = len(a) // 2
        left = self.mergesort(a[:mid])
        right = self.mergesort(a[mid:])
        
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        
        return np.array(result)
    
    def makeUnionFind(self, N):
        return {i: np.array([i]) for i in range(N)}
    
    def find(self, u, v):
        return u[v][0]
    
    def union(self, u, s1, s2):
        for key in u:
            if self.find(u, key) == s1:
                u[key][0] = s2
        return u


# Testing the example
A = np.array([[0, 8, 0, 3], [0, 0, 2, 5], [0, 0, 0, 6], [0, 0, 0, 0]])
obj = kruskalClass()
T = obj.findMinimumSpanningTree(A)
print(T)
