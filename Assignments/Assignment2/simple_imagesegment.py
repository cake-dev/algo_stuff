# Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from SegmentationClass import SegmentationClass

# load image into numpy array with pillow
image_file_url = 'TestPics/test_Twist.png'
image_file = Image.open(image_file_url)
image_file = np.asarray(image_file).astype(np.int32)
# keep only the first 3 channels (RGB)
image_file = image_file[:, :, :3]

image = image_file

# Example usage
ff_segment_class = SegmentationClass()
ff_segment_class.set_hyperparameters(p0=10, x_a=[0,0], x_b=[4,5])

A = ff_segment_class.construct_adjacency_matrix(image)

segmenting_mask = ff_segment_class.segment_image(image)

# Plot the results
fig, axs = plt.subplots(1,2)
fig.suptitle(image_file_url)
axs[0].imshow(image.astype(np.uint8), interpolation='nearest')
axs[0].set_title("Input image")
# The matrix 'segmenting_mask' is binary, but it is helpful to scale the values to be 0 or 255
#  when displaying with imshow
axs[1].imshow(255*segmenting_mask.astype(np.uint8), interpolation='nearest')
# set colors to be dark grey and light grey
axs[1].set_title("Binary segmentation")
# show filename on plot
# plt.text(0.5, 0.04, image_file_url, color='r', fontsize=12, ha='center')
# show foreground and background points on plot
plt.plot(ff_segment_class.x_a[1], ff_segment_class.x_a[0], 'ro')
plt.plot(ff_segment_class.x_b[1], ff_segment_class.x_b[0], 'bo')
plt.show()