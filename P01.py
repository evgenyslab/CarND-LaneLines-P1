#importing some useful packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys

print sys.prefix

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize_image(img):
    """ Normalizes a grayscale image"""
    normalizedImg = np.zeros_like(img)
    cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., lmbda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + Lambda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lmbda)

import os
os.listdir("test_images/")

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Lane tracker is an object that tracks both left and right lanes.
# It processes images, runs lane detection pipe-line, and makes an estimate as to
# which lines correspond to Left/Right lanes.
# Lanes are stored as (m,b) line variables with 'soft' tracking enabled such that
# position is filtered to ensure smoothness
def laneTracker():
    def self():

        pass


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Create copy of Image
    output_image = image.copy()
    # Convert image to Gray:
    output_image = grayscale(output_image)
    # Normalize the image: This will increase contrast under certain conditions and may improve edge detection (unless canny function has this built in)
    output_image = normalize_image(output_image)
    # Apply Gaussian Blur to image:
    output_image = gaussian_blur(output_image,7)
    # Run Canny Edge detector:
    edges = canny(output_image, 50, 150)
    # Apply image mask:
    imshape = image.shape
    crops = [7.9/16.0, 8.1/16.0, 9/16.0]
    vertices = np.array([[(0,imshape[0]),(crops[0]*imshape[1], crops[2]*imshape[0]), (crops[1]*imshape[1], crops[2]*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    output_image = region_of_interest(edges,vertices)
    # run Hough Lines
    #     line_image = np.copy(image)*0
    #     lines = hough_lines(output_image, 1, np.pi/180, 50, 100, 3)
    #     for line in lines:
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    #     # Create a "color" binary image to combine with line image
    #     color_edges = np.dstack((edges, edges, edges))

    #     # Draw the lines on the edge image
    #     lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    cv2.imshow("test",output_image)
    cv2.waitKey(100)
    return output_image


# fileName = 'solidWhiteRight.mp4'
# fileName = 'solidYellowLeft.mp4'
fileName = 'challenge.mp4'
white_output = 'test_videos_output/' + fileName
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip('test_videos/' + fileName)
count =1
for frames in clip1.iter_frames():
    process_image(frames)
    # cv2.imshow("test",frames)
    # cv2.waitKey(100)
    count+=1
print count

# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
cv2.destroyAllWindows()