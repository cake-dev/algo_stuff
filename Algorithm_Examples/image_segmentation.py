# python program for graph based image segmentation using min cut / max flow algorithm

import numpy as np
import cv2
import networkx as nx

# read image
img = cv2.imread('../data/img/download.jpeg')
print(img.shape)
# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# get dimensions of image
nrows, ncols, nch = img.shape

# create graph
g = nx.DiGraph()

# add pixels as nodes
nodeids = np.arange(nrows * ncols).reshape((nrows, ncols))
g.add_nodes_from(nodeids.ravel())

# add edges between pixels
# horizontal and vertical edges
weights = np.linalg.norm(img[1:, :] - img[:-1, :], axis=2)
weights = np.vstack((weights, np.zeros((1, ncols))))
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), nodeids[1:, :].ravel(), weights))
g.add_weighted_edges_from(edges)

weights = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=2)
weights = np.hstack((weights, np.zeros((nrows, 1))))
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), nodeids[:, 1:].ravel(), weights))
g.add_weighted_edges_from(edges)

# diagonal edges
weights = np.linalg.norm(img[1:, 1:] - img[:-1, :-1], axis=2)
weights = np.vstack((weights, np.zeros((1, ncols))))
weights = np.hstack((weights, np.zeros((nrows, 1))))
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), nodeids[1:, 1:].ravel(), weights))
g.add_weighted_edges_from(edges)

weights = np.linalg.norm(img[1:, :-1] - img[:-1, 1:], axis=2)
weights = np.vstack((weights, np.zeros((1, ncols))))
weights = np.hstack((np.zeros((nrows, 1)), weights))
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), nodeids[1:, :-1].ravel(), weights))
g.add_weighted_edges_from(edges)

# add terminal edges
# add source edges
g.add_weighted_edges_from([(nodeids[0, 0], 's', 0)])
g.add_weighted_edges_from([(nodeids[-1, -1], 's', 0)])

# add sink edges
g.add_weighted_edges_from([(nodeids[0, 0], 't', 0)])
g.add_weighted_edges_from([(nodeids[-1, -1], 't', 0)])

# add terminal edge weights
weights = np.linalg.norm(img - img[0, 0], axis=2)
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), ['s'] * (nrows * ncols), weights))
g.add_weighted_edges_from(edges)

weights = np.linalg.norm(img - img[-1, -1], axis=2)
weights = weights.ravel()
edges = list(zip(nodeids.ravel(), ['t'] * (nrows * ncols), weights))
g.add_weighted_edges_from(edges)

# find min cut
cut_value, partition = nx.minimum_cut(g, 's', 't')
reachable, non_reachable = partition

# create mask
mask = np.zeros((nrows, ncols), dtype=np.uint8)
mask.ravel()[list(reachable)] = 1

# apply mask
img2 = img * mask[:, :, np.newaxis]

# display image
cv2.imshow('Original Image', img)
cv2.imshow('Segmented Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# References:
# https://www.youtube.com/watch?v=JtXvUoPx4Qs