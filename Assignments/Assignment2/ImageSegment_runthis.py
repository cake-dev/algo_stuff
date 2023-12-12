# Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from SegmentationClass import SegmentationClass
import glob
import matplotlib.pyplot as plt

image_dir = 'TestPics/'
image_files = glob.glob(image_dir + '*.png')

# Show images and allow user to select one, save the filepath of the image to a variable
import os
import tkinter as tk
from PIL import Image, ImageTk

def select_image(directory):
    # SOURCE: TKinter Documentation
    # Create a new tkinter window
    window = tk.Tk()

    # Variable to store the selected image path
    selected_image_path = tk.StringVar()

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Loop through the files
    for file in files:
        # Construct the full file path
        filepath = os.path.join(directory, file)

        # Open and resize the image
        image = Image.open(filepath)
        image = image.resize((100, 100), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        # Create a button for the image
        button = tk.Button(window, image=photo, command=lambda f=filepath: (selected_image_path.set(f), window.destroy()))
        button.image = photo  # Keep a reference to the image to prevent it from being garbage collected
        button.pack()

    # Start the tkinter main loop
    window.mainloop()

    # Return the selected image path
    return selected_image_path.get()

# Use the function
image_path = select_image(image_dir)
print('Selected image:', image_path)

# load image into numpy array with pillow
image_file_url = image_path
image_file = Image.open(image_file_url)
image_file = np.asarray(image_file).astype(np.int32)
# keep only the first 3 channels (RGB)
image_file = image_file[:, :, :3]

image = image_file

def manual_select_foreground_background(image):
    fig = plt.figure()
    fig.suptitle('Select Foreground and Background Points')
    ax = fig.add_subplot(111)
    ax.imshow(image.astype(np.uint8), interpolation='nearest')

    xc = []
    yc = []
    print('1) Click on start and end point of the desired profile')
    print('2) Close the figure to continue the profile plotting')

    def onclick(event):
        if event.button == 1 and len(xc) < 2:  # Only accept two clicks
            xcc, ycc = event.xdata, event.ydata
            # round to nearest 1
            xcc = round(xcc)
            ycc = round(ycc)
            xc.append(xcc)
            yc.append(ycc)
            print('({}, {})'.format(xcc, ycc))
            ax.plot(xcc, ycc, 'ro')
            ax.figure.canvas.draw()  # Update the figure with the new point

        if len(xc) == 2:  # If two points have been selected, disconnect the event and close the figure
            fig.canvas.mpl_disconnect(cid)
            plt.close()

    cid = fig.canvas.mpl_connect('button_release_event', onclick)
    plt.show()

    start_yx = [yc[0], xc[0]]
    end_yx = [yc[1], xc[1]]
    # round to nearest 1
    return start_yx, end_yx

x_a_point, x_b_point = manual_select_foreground_background(image)
print('x_a_point: ', x_a_point)
print('x_b_point: ', x_b_point)

# Example usage
ff_segment_class = SegmentationClass()
ff_segment_class.set_hyperparameters(p0=10, x_a=x_a_point, x_b=x_b_point) # MANUALLY ENTER POINT HERE IF GUI NOT WORKING

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