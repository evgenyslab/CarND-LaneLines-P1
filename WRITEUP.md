# **Finding Lane Lines on the Road** 
Author: Evgeny Nuger.
Date: November 9<sup>th</sup>, 2017


## Overview
The purpose of this exercise was to develop a computer-vision pipeline for lane detection using Hough-lines.


The proposed pipe-line consists of two major steps:
* Hough line detection
* Estimation and tracking of left and right lanes

The pipeline was tested for the two provided videos and the challenge video. The pipeline works on all three videos. Results are provided in the ```test_videos_output``` folder. To re-generate the result videos, re-run ```P01.py``` with the desired video uncommented.

## Pipeline

### Hough Line Detection
The Hough lines were extracted through a process similar to the tutorials provided in the course.

Firstly, the input image was converted to gray scale, then normalized, and then a Gaussian blur was applied. The purpose of the normalization was to increase the contrast in the grey scale image. Ideally, increasing the contrast could result in improved edge detection (in the case that the Canny edge detection algorithm does not prodive a normalization step). A Gaussian blur with a 7-pixel kernel size was implemented to smooth each image and remove high frequency noise that may lead to incorrect edge detections.

Following the pre-processing the image was cropped with a trapezoidal cropping region that was estimated based on the input video feed. This static region worked well for the provided videos, but may not work well in tight corners.

The Hough line detection uses fixed parameters to recover line values. The parameters were chosen based on experimentation with the input training videos.


### Estimation and Tracking of Left and Right Lanes
Lane estimation is handled in two parts: Left/Right lane segmentation and detection, and a multi-frame lane smoothing filter. The segmentation of the Left/right lanes occurs in the function: ```process_lines(self,lines)```. This function takes in the lines returned by the Hough lines function, estimates the slopes and y-intercepts of the point-pairs on the image, and removes all point pairs with an absolute angle less than 30&deg; to the x-axis. This is a soft assumption and may not work in tight corner cases. 

Once a list of filtered point pairs is created, the list is separated into two categories: Left, Right, based on whether the angle of the line is greater than or less than 0&deg;. The average slope and intercept point are taken for all lines in each Left/Right category to produce the line estimate for the current frame. The minimum y-coordinate on the image of each line was recorded as well, but only the first value is used (earlier implementations dynamically adjusted the minimum y-value, i.e. the furtherest point to draw the lane marker, but this was later abandoned).

The average slopes and intercepts for each line were then parsed to an averaging filter to estimate the current lane position and orientation from a 5-frame history. A weighed average filter was implemented on the previous 5 frames to smooth the motion of lane detection. If no lane was detected, the tracker simply pushes the last measurement to the current estimate similar to how a Kalman filter would operate. The averaging filter was setup to avoid ambiguity during initialization (i.e. when there are less than 5 measurements) by dynamically updating the vecotry of weights based on the number of available measurements in the 5-frame queue.

The length of drawing vectors are determined by the first frame (this is fallible to errors if first lanes detected are very short). To combat potential pitfalls, the length of the lane is set to the maximum detected lane (left or right), and applied to both. To keep lane-guide length consistent throughout the video, the same length is used throughout the video.

Once the lanes have been detected, and segemented into Left/Right, the lanes are drawn onto the image with different colors for each side.

## Notes on program


The current implementation of the program makes several 'soft' assumptions about how lane detection will work. This includes keeping 'lost' lanes in the same location as they were last detected, and that lanes are straight (no curvature handled). Due to these assumptions, there are some limitations as to the ability to estimate lane position and orientation when they are not detected. The current implementation simply keeps the last known position until a new position is available.


### Future Improvements

The following are potentional improvements to the algorithms and suggestions for different approaches:
1. Automated Hough line parameters
    * Parameterizing and dynamically modifying the Hough line parameters could improve line detection under various conditions. Would require a controller that analyzes the image and determines which parameters to update. Could be potentially easier to implement with neural networks instead.
2. Adaptive ROI cropping
    * The current Region of Interest (ROI) is statically set. It could be modified online based on the lanes detected. This could help overcome potential issues around tight corners.
3. Recovering Lane Curvature
    * Straight lines provide very limited information about the road, in contrast, curved lanes could provide more information about the roadway. Recovering the curvature could be done by estimating splines instead of lines. This would make the recovered more useful for other downstream processing.
4. Implementation of a filtering algorithm for lane tracking
    * A filtering algorithm, or probabilistic approach to estimating the lane states (i.e.: slope, intercept, R-&theta;, etc...) could reduce the positioning error of the lanes. Further, it could provide a more accurate approach to estimating lane position when the lane is not detected. 
5. Neural Network lane detection
    * Neural nets, or CNNs more specifically, could in theory work better across a wider range of scenes for lane detection, given that enough variance exists in training data. CNN's could encode the improvements in #1, 2 and 5. Further, curved lanes could be extracted as well with correct training data.