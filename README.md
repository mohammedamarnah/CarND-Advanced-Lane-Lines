## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Advanced Lane Finding Project - Udacity SDCN

### Mohammed Amarnah

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./README-images/Originalcal.png "Original Chessboard"
[image2]: ./README-images/Processedcal.png "Undistorted Chessboard"
[image3]: ./README-images/download.png "Original Image"
[image4]: ./README-images/download-1.png "Different Gradient Thresholding methods"
[image5]: ./README-images/download-2.png "Color Thresholding(left), and Combined Thresholding(right)"
[image6]: ./README-images/download-3.png "Perspective Transformation"
[image7]: ./README-images/download-4.png "Histogram"
[image8]: ./README-images/download-5.png "Line Detection"
[image9]: ./README-images/download-6.png "Final Pipeline (shows polynomial fitting)"
[video1]: ./README-images/project_video_output.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration and Distortion Correction

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cells in my notebook file. The first of them was for function definition, and the second one was to test the functions and see output.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objectpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imagepoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objectpoints` and `imagepoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Color/Gradient Threshold

#### 1. Gradient Thresholding
In this step, I tried applying different gradient thresholding methods on a random image from my testing set. I tried Sobel on the x-axis and the y-axis, and then I tried the magnitude threshold, and then the direction of the gradient threshold. 
Finally I tried a combined (gradient only) threshold. Here are some of the results that I got in each step:

![alt text][image3]

![alt text][image4]

#### 2. Color Thresholding (+ Combined Final Threshold)
Next, I tried applying color thresholds on images. And after that, I applied a threshold that combines color and gradient thresholds in one image to create a binary image. Here are the results I got:

![alt text][image5]

### Perspective Transformation
The code for my perspective transform includes a function called `transform_perspective()`, which appears under the "Perspective Transform" cell. The `transform_perspective()` function takes as inputs an image (`img`), and applies the transform using the opencv functions `getPerspectiveTransform()` and `warpPerspective()`. I chose the hardcoded source and destination points by testing manually. It was hard, but I got good results. Here's an image after perspective transformation:

![alt text][image6]

### Lane Lines Detection

#### 1. Creating a histogram peaks image

In this step, I created a histogram from the outputted image from the transformation step. It was made to be used by the sliding window algorithm later. Here's a sample for a histogram:

![alt text][image7]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After finding the histogram, I applied the sliding window algorithm to identify the lanes in the image (the peaks in the histogram). You'll find two functions that does the same thing, the first one was just to test the algorithm on a single image and it was called `findLines_test()`. I made sure the algorithm is working correctly and that it fits the lines well. Here's a sample of the output I got: 

![alt text][image8] 

#### 3. Curvature calculation and final touches

Finally, I implemented the `Line()` class to keep track of the last detected line (that increases the accuracy alot!) and some other helper functions. One of the functions was the `findCurv()` that calculates the curvature for a given set of points. I also implemented a `drawPolynomial()` function that draws the calculated result on an image, and writes the curvature and the distance from the center.

#### 4. Putting Everything together (Video)

In the final step, I implemented a function that takes an image, applies all of the above and then returns the processed image. Here's a sample of what I got from a single image:

![alt text][image9]

And then finally, I applied the pipeline on the project video and here's the result I got:

[link to my video result](./README images/project_video_output.mp4)

Here's a link for the video on youtube:
https://youtu.be/0Z_x0EgV8HU

UPD: I tried the pipeline on the challenge video and it's working well. I'll update again after trying it on the harder challenge video! :D .

---

### Discussion

The first step I took was to calibrate the camera, after that I applied the distortion correction on images. The next step was to apply some thresholds; what I did was apply some combined threshold (Color and Gradient). I used Sobel (on both axes), Magnitude of the gradient, Direction of the gradient, and Color thresholds. 
After that, I applied a perspective transform on the images, the next step from that was to detect the lane lines. I did that using the sliding window algorithm (by identifying the peaks in a histogram).

Obviously, this pipeline fails in extreme whether conditions, where the road might not be clear (like snow, rain, or sun reflections). What I plan on doing is converting this pipeline into a whole new one based on a research paper by a team from Oxford University titled: 
`Weakly Supervised Segmentation of Path Proposals`
and here's a link for the paper: 
https://arxiv.org/abs/1610.01238
