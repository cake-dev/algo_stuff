# AUTHOR: Jake Bova
# DATE LAST MODIFIED: 11/04/2023
# DESCRIPTION: This file contains the code for the MST Kruskals assignment.
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class kruskalClass:
    
    def __init__(self):
        pass  # Skipping initialization for brevity

    def plot_graph(self, edges, n, pos, highlighted_edges=[]):
        """
        Plots a graph given the edges and the number of nodes.
        """
        G = nx.Graph()  # Creating a new graph
        G.add_nodes_from(range(n))  # Adding nodes
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)  # Adding edges with weights
        
        nx.draw_networkx_edges(G, pos, alpha=0.5, style="dashed")  # Drawing edges

        if highlighted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, alpha=0.8, width=2, edge_color="b")  # Highlighting specific edges

        nx.draw_networkx_nodes(G, pos)  # Drawing nodes
        nx.draw_networkx_labels(G, pos)  # Adding labels to nodes
        edge_labels = {(u, v): w for u, v, w in edges}  # Creating edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # Adding labels to edges

        plt.show()  # Displaying the graph

    def mergesort(self, a):
        """
        Performs the merge sort algorithm on a given numpy array 'a'. Returns
        the sorted array along with the indices of the elements as per their 
        order in the original array.

        - Input: a must be a 1xK numpy array.
        - Output: b must be a 1xK numpy array with elements in ascending order (i.e.,
            lowest value to highest).

        Parameters:
        - a : numpy array
            The input array to be sorted.

        Returns:
        - sorted_array : numpy array
            The array 'a' sorted in ascending order.
        - sorted_inds : numpy array
            The indices of elements in the original array corresponding to the order in sorted_array.
        """
        if len(a) <= 1:
            return a, np.arange(len(a))  # Returning the original array and indices if the length is less than or equal to 1
        
        mid = len(a) // 2  # Calculating the middle index for splitting
        left, left_inds = self.mergesort(a[:mid])  # Sorting the left half
        right, right_inds = self.mergesort(a[mid:])  # Sorting the right half
        right_inds += mid  # Adjusting indices for the right half
        
        sorted_array, sorted_inds = self.merge(left, left_inds, right, right_inds)  # Merging the sorted halves
        return sorted_array, sorted_inds  # Returning the sorted array and indices
    
    def merge(self, left, left_inds, right, right_inds):
        """
        Merges two sorted arrays while maintaining the order. Used as a 
        helper function for the mergesort algorithm.

        Parameters:
        - left, right : numpy array
            The sorted arrays to be merged.
        - left_inds, right_inds : numpy array
            The indices of elements in the original unsorted array.

        Returns:
        - sorted_array : numpy array
            The merged sorted array.
        - sorted_inds : numpy array
            The indices of elements corresponding to the order in sorted_array.
        """
        i = j = 0  # Initializing indices
        sorted_array = np.zeros(len(left) + len(right))  # Creating an array for sorted elements
        sorted_inds = np.zeros(len(left) + len(right), dtype=int)  # Creating an array for indices of sorted elements
        
        # Merging process while both halves have elements
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                sorted_array[i+j] = left[i]  # Adding the smaller element to the sorted array
                sorted_inds[i+j] = left_inds[i]  # Adding the index of the smaller element
                i += 1  # Incrementing the index for the left half
            else:
                sorted_array[i+j] = right[j]  # Adding the smaller element to the sorted array
                sorted_inds[i+j] = right_inds[j]  # Adding the index of the smaller element
                j += 1  # Incrementing the index for the right half
        
        # Adding remaining elements from the left half, if any
        while i < len(left):
            sorted_array[i+j] = left[i]  # Adding element to the sorted array
            sorted_inds[i+j] = left_inds[i]  # Adding the index of the element
            i += 1  # Incrementing the index

        # Adding remaining elements from the right half, if any
        while j < len(right):
            sorted_array[i+j] = right[j]  # Adding element to the sorted array
            sorted_inds[i+j] = right_inds[j]  # Adding the index of the element
            j += 1  # Incrementing the index
        
        return sorted_array, sorted_inds  # Returning the sorted array and indices

    def makeUnionFind(self, N):
        """
        Creates and returns a union-find data structure for N nodes.

        Parameters:
        - N : int
            The total number of nodes.

        Returns:
        - u : dictionary
            The union-find data structure with N nodes.
        """
        u = {}  # Initializing the union-find data structure
        for i in range(N):
            u[i] = np.array([i, 1])  # Each node is its own parent has a count of 1
        return u  # return a union-find data structure with N nodes
    
    def find(self, u, v):
        """
        Finds and returns the representative element or the "parent" of the set 
        to which the element 'v' belongs in the union-find data structure 'u'.

        Parameters:
        - u : dictionary
            The union-find data structure.
        - v : int
            The element whose set representative is to be found.

        Returns:
        - s : int
            The representative element of the set to which 'v' belongs.
        """
        return u[v][0]  # Finding the set of the node v
    
    def union(self, u, s1, s2):
        """
        Unions or merges two distinct sets s1 and s2 in the union-find data 
        structure 'u'. The representative of the merged set becomes the representative 
        of set s1.

        Parameters:
        - u : dictionary
            The union-find data structure.
        - s1, s2 : int
            The representative elements of the two sets to be merged.

        Returns:
        - u : dictionary
            The updated union-find data structure after the union operation.
        """
        # Check sizes of s1 and s2 to ensure correct representative
        if u[s1][1] < u[s2][1] or (u[s1][1] == u[s2][1] and s1 > s2): # Choose larger set or smaller representative if sizes are equal
            s1, s2 = s2, s1
        for key in u:
            if u[key][0] == s2:
                u[key][0] = s1  # Merging the sets
        u[s1][1] += u[s2][1] # Updating the size of the merged set
        return u  # Returning the updated union-find structure
    
    def findMinimumSpanningTree(self, A):
        """
        Implements Kruskal's algorithm to find and return the minimum 
        spanning tree of a graph represented by the adjacency matrix 'A'.

        Parameters:
        - A : numpy array
            The adjacency matrix representing the graph.

        Returns:
        - T : numpy array
            The adjacency matrix representing the minimum spanning tree.
        """
        N = len(A)  # Getting the number of nodes
        # Behold, the power of list comprehension!
        edges = [(A[i][j], i, j) for i in range(N) for j in range(i+1, N) if A[i][j] > 0]  # Extracting edges and weights from the adjacency matrix (A[i][j] corresponds to the weight of the edge between nodes i and j, this builds a list of tuples (weight, node1, node2)
        edge_weights = np.array([x[0] for x in edges])  # Getting the weights of edges (x[0] corresponds to the weight of the edge)
        sorted_edge_weights, sorted_inds = self.mergesort(edge_weights)  # Sorting the edges based on weights
        sorted_edges = [edges[i] for i in sorted_inds]  # Getting the sorted edges from esges using the indices
        
        uf = self.makeUnionFind(N)  # Creating a union-find data structure for N nodes (this is used to check for cycles)
        T = np.zeros((N, N))  # Creating an array for the minimum spanning tree
        
        for edge in sorted_edges:
            weight, node1, node2 = edge  # Extracting weight and nodes from the edge
            set1, set2 = self.find(uf, node1), self.find(uf, node2)  # Finding the sets of the nodes
            
            if set1 != set2:
                T[node1][node2] = weight  # Adding the edge to the minimum spanning tree if it does not form a cycle
                uf = self.union(uf, set1, set2)  # Merging the sets
        
        return T  # Returning the minimum spanning tree
    
    def plotMinimumSpanningTree(self, A):
        """
        Plots the minimum spanning tree of a graph represented by the adjacency matrix 'A'.
        """
        T = self.findMinimumSpanningTree(A)
        N = len(A)
        edges = [(i, j, A[i][j]) for i in range(N) for j in range(i+1, N) if A[i][j] > 0] # Extracting edges and weights from the adjacency matrix of A
        mst_edges = [(i, j, T[i][j]) for i in range(N) for j in range(i+1, N) if T[i][j] > 0] # Extracting edges and weights from the adjacency matrix of T
        
        G = nx.Graph() # Creating a new graph
        G.add_nodes_from(range(N))
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        pos = nx.spring_layout(G)  # Compute positions only once so they don't change on every iteration
        
        print("Original Graph:")
        self.plot_graph(edges, N, pos)
        
        highlighted = [] # Add highlighted edges to show the MST
        for edge in mst_edges:
            highlighted.append(edge)
            print(f"Adding Edge: {edge}")
            self.plot_graph(edges, N, pos, highlighted)