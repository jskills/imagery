import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

image_type = None
file_or_dir = None
dir_flag = False
args_error = False
output_dir = "output/"


###

def process_image(image_file):

    # load and plot image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")

    if DEBUG:
        plt.title("Original Image")
        filebody = os.path.basename(image_file)
        extension = os.path.splitext(image_file)[1]
        filebody = filebody.replace(extension,'')
        output_file = "".join((output_dir, filebody, extension))
        print(output_file)
        plt.savefig(output_file)
        plt.show()


    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    plt.imshow(gray)

    # get edged image
    edges = cv2.Laplacian(gray, -1, ksize=5)
    edges = 255 - edges
    ret, edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    plt.imshow(edges)

    # blur images heavily using edgePreservingFilter
    edgePreservingImage = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)

    # create output matrix
    cartoon = np.zeros(gray.shape)
    #combine cartoon image and edges image
    cartoon = cv2.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edges)
    cartoon = cv2.stylization(img, sigma_s=150, sigma_r=0.25)

    plt.imshow(cartoon)
    if DEBUG:
        plt.title("Cartoon Image 1")

    filebody = os.path.basename(image_file)
    extension = os.path.splitext(image_file)[1]
    filebody = filebody.replace(extension,'')
    output_file = "".join((output_dir, filebody, '-cartoon1', extension))
    print(output_file)
    plt.savefig(output_file)
    plt.show()

    cartoon_image1, cartoon_image2  = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.5, shade_factor=0.02)
    plt.imshow(cartoon_image2)
    if DEBUG:
        plt.title("Cartoon Image 2")

    filebody = os.path.basename(image_file)
    extension = os.path.splitext(image_file)[1]
    filebody = filebody.replace(extension,'')
    output_file = "".join((output_dir, filebody, '-cartoon2', extension))
    print(output_file)
    plt.savefig(output_file)
    plt.show()

###

if sys.argv[1:]:
    file_or_dir = sys.argv[1]
    if os.path.isdir(file_or_dir):
        dir_flag = True
    elif os.path.isfile(file_or_dir):
        dir_flag = False
    else:
        args_error = True
else:
    args_error = True

if args_error:
    print("Usage : cartoon.py [file_name|dir_name]")
    sys.exit()

DEBUG = True

###

if dir_flag:
    # pull all files in dir and process
    for filename in os.listdir(file_or_dir):
        full_path = "".join((file_or_dir,'/', filename))
        print(full_path)
        process_image(full_path)
else:
    process_image(file_or_dir)


