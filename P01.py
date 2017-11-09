import sys
import os
import math
import numpy as np
import cv2
from moviepy.editor import VideoFileClip


class tracker():
    """
    This is a smoothing tracker class to track left/right lanes individually.

    """
    def __init__(self):
        # define all local variables
        self.m = None
        self.b = None
        self.ymin = None
        self.m_prev = None
        self.b_prev = None
        self.ymin_prev = None
        self.smoothing = (0.75, 0.25)
        self.smoothingy = (0.55, 0.45)
        self.weights = np.array([0.45, 0.25, 0.15, 0.1, 0.05])


        self.marr = np.array([None, None, None, None, None])
        self.barr = np.array([None, None, None, None, None])

    def update(self,slope,intc,ym):
        # circ shift tracking buffer:
        self.marr = np.roll(self.marr.copy(),1)
        self.barr = np.roll(self.barr.copy(),1)
        # updates lane position based on whether lane was detected (simple tracker)
        if np.any(np.isnan((slope,intc,ym))):
            # no valid input, keep previous estimate...
            # shift current measurement to previous, update with smoothing:
            self.m_prev = self.m
            self.b_prev = self.b
            self.marr[0] = self.marr[1]
            self.barr[0] = self.barr[1]
            self.ymin_prev = self.ymin
        else:
            # shift current measurement to previous, update with smoothing:
            self.m_prev = self.m
            self.b_prev = self.b
            self.ymin_prev = self.ymin

            self.marr[0] = slope
            self.barr[0] = intc
            # first time initialization
            if None in (self.m_prev, self.b_prev, self.ymin_prev):
                self.m = slope
                self.b = intc
                self.ymin = ym

            else:
                self.m = self.smoothing[0]*slope + self.smoothing[1]*self.m_prev
                self.b = self.smoothing[0]*intc  + self.smoothing[1]*self.b_prev
                # self.ymin = self.smoothing[0]*ym + self.smoothing[1]*self.ymin_prev
                # ALTERNATIVE:
                newWeights = self.weights[self.marr!=None]/self.weights[self.marr!=None].sum()
                self.m = np.dot(self.marr[self.marr!=None],newWeights)
                self.b = np.dot(self.barr[self.marr!=None],newWeights)


    def get_points(self,imsize):
        # returns start/end points of lane to draw
        # calculate the end points using slope y-intercept.
        p1 = (int((self.ymin-self.b)/self.m),int(self.ymin))
        p2 = (int((imsize[0]-self.b)/self.m),imsize[0])
        return p1, p2


class lane_detector():
    """
    Lane tracker is an object that tracks both left and right lanes.
    It processes images, runs lane detection pipe-line, and makes an estimate as to
    which lines correspond to Left/Right lanes.
    Lanes are stored as (m,b) line variables with 'soft' tracking enabled such that
    position is filtered to ensure smoothness

    """
    def __init__(self):
        self.r_lane = tracker()
        self.l_lane = tracker()
        pass

    def grayscale(self,img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self,img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self,img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def normalize_image(self,img):
        """ Normalizes a grayscale image"""
        normalizedImg = np.zeros_like(img)
        cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        return normalizedImg

    def region_of_interest(self,img, vertices):
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

    def process(self,image):
        # apply all preprocessing in one step:
        output_image = self.gaussian_blur(self.normalize_image(self.grayscale(image.copy())),7)
        # Run Canny Edge detector:
        edges = self.canny(output_image, 50, 150)
        # Apply image mask:
        imshape = image.shape
        # arbitrarily set polygon crop region, seems to work
        crops = [7.9/16.0, 8.1/16.0, 9/16.0]
        vertices = np.array([[(0,imshape[0]),(crops[0]*imshape[1], crops[2]*imshape[0]), (crops[1]*imshape[1], crops[2]*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
        output_image = self.region_of_interest(edges,vertices)
        # Get Hough lines:
        lines = cv2.HoughLinesP(output_image, 1, np.pi/180, 50, np.array([]), 15, 3)
        # now, process lines to get estimate of best fit lanes:
        # Color convertion for opencv:
        image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        line_image = np.zeros_like(image)
        try:
            self.process_lines(lines)
            p1_L, p2_L = self.l_lane.get_points(imshape)
            p1_R, p2_R = self.r_lane.get_points(imshape)
            cv2.line(line_image, p1_L, p2_L, [255,0,0], 6)
            cv2.line(line_image, p1_R, p2_R, [0,255,0], 6)
        except:
            pass


        alpha_image = cv2.addWeighted(image.copy(), 0.8, line_image, 1.0, 0.)

        cv2.imshow("test",alpha_image)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # convert color back:
        return cv2.cvtColor(alpha_image.copy(), cv2.COLOR_BGR2RGB)

    def process_lines(self,lines):
        # for each line in lines, group them by their angle, can make soft assumption that positive slopes will correspond to right lanes, negative slopes - left lanes (since y-coords are flipped)
        # upgrade array to float:
        flines = lines.astype(float)
        linesLocal = flines.copy()
        # get slopes:
        slopes = np.divide((flines[:,:,3]-flines[:,:,1]),(flines[:,:,2]-flines[:,:,0]))
        # get y-intercepts:
        intcpt = flines[:,:,1] - np.multiply(slopes,flines[:,:,0])
        # get line angles to x-line:
        thetas = np.arctan2((flines[:,:,3]-flines[:,:,1]),(flines[:,:,2]-flines[:,:,0]))
        # get lines to keep:
        keeps = abs(thetas)>30*math.pi/180 # THIS IS IMPORTANT FOR CHALLENGE!
        # remove bad data lines:
        thetas = thetas[keeps.reshape([keeps.size]),:]
        intcpt = intcpt[keeps.reshape([keeps.size]),:]
        slopes = slopes[keeps.reshape([keeps.size]),:]
        linesLocal = linesLocal[keeps.reshape([keeps.size]),:,:]
        # find furthest line point (get min y-value):
        ymin = np.array([linesLocal[:,:,1],linesLocal[:,:,3]]).min()
        # segment lines into L/R:
        left = thetas<0
        right = thetas>0
        # get average slope & intercept for both sides
        l_slope, l_int = self.get_mean_line(slopes[left.reshape([left.size]),:],intcpt[left.reshape([left.size]),:])
        r_slope, r_int = self.get_mean_line(slopes[right.reshape([right.size]),:],intcpt[right.reshape([right.size]),:])
        # filter to slopes and ints for L/R lanes:
        self.r_lane.update(r_slope,r_int,ymin)
        self.l_lane.update(l_slope,l_int,ymin)


    def get_mean_line(self,slopes=np.empty(1),intercepts=np.empty(1)):
        return slopes.mean(), intercepts.mean()



def main():
    """
    Function handler, loads data, sends to processing.
    """

    # fileName = 'solidWhiteRight.mp4'
    fileName = 'solidYellowLeft.mp4'
    # fileName = 'challenge.mp4'
    white_output = 'test_videos_output/' + fileName
    clip1 = VideoFileClip('test_videos/' + fileName)
    # make lane tracker object:
    detector = lane_detector()
    white_clip = clip1.fl_image(detector.process)
    white_clip.write_videofile(white_output, audio=False)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()