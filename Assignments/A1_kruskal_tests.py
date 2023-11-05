import numpy as np
from MST import kruskalClass


# Testing
# obj = kruskalClass()
# A = np.array([[0, 1, 3, 0], [0, 0, 2, 4], [0, 0, 0, 5], [0, 0, 0, 0]])
# T = obj.findMinimumSpanningTree(A)
# print(T)

# a = np.array([5, 2, 9, 1, 5, 6])
# b, inds = obj.mergesort(a)
# print(b)
# print(inds)

# Testing
obj = kruskalClass()
A = np.array([[0, 1, 3, 0], [0, 0, 2, 4], [0, 0, 0, 5], [0, 0, 0, 0]])
obj.plotMinimumSpanningTree(A)

# #Instantiate an object for your class.
# obj = kruskalClass()
# # Create a test matrix
# A = np.array([[0, 8, 0, 3],
#  [0, 0, 2, 5],
#  [0, 0, 0, 6],
#  [0, 0, 0, 0]])
# #Use code to generate a MST
# T = obj.findMinimumSpanningTree(A)
# #Print the MST
# print(T)
# print(type(T))

# n = 5; 
# u = obj.makeUnionFind(n)
# print(u)

# # If we run the 'find' it returns the index 
# # that was provided as input
# s1 = obj.find(u,2)
# print(s1)
# s2 = obj.find(u,4)
# print(s2)

# # Now we can try doing some union operations
# #Combine the sets for nodes 0 and 1
# u1 = obj.union(u,obj.find(u,0),obj.find(u,1))
# print(obj.find(u1,0))
# print(u1)
# u2 = obj.union(u,obj.find(u1,0),obj.find(u1,2))
# print(u2)

# #Notice that the set '2' takes the name of the larger set, 
# # which is composed of {0,1} from the first merging operation.  
# # When doing the second union operation, your code should always give node '2' 
# # the name obj.find(0) (which may be '1' or '0' depending upon your implementation) 
# # because it is the larger set 