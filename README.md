# Handy
Handy is a hand detection software written in C++ using OpenCV v3.4.1. The software is capable of recognizing hands in an video and of counting the number of lifted fingers.

## Software assumptions
A few assumptions have been made:
1. The camera is supposed to be static.
2. The camera has no automatic regulations, such as auto-focus etc.
3. The user is not moving in the frame (eg: he sits at his desk in front of the camera).
4. There are no particular constraints on the color of the background, but it should be approximately static (no moving objects/strong changes of illumination in the background).

We decided not to add invasive constraints such as forcing the user to wear gloves, in order to change the color of his hand, or to have a specific illumination in the scene.

A demo can be watched [here](https://www.youtube.com/watch?v=z8rWGQyIQAE).

![gif demo](https://j.gifs.com/pQmJ8m.gif)

## Usage
This software has been developed on Visual Studio 14 using OpenCV v3.4.1. If you are using the same development environment you can simply clone the project and open it in Visual Studio.

Remember to create an environment variable called `OPENCV_PATH` that points to the build folder of OpenCV (eg: `C:\Users\UserName\Documents\opencv\build`). Otherwise you will need to update the properties of the project so that it knows where OpenCV is on your machine.

#### Camera configuration
After running the program (just run the [main](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/main.cpp)) you will be presented with a window where you can configure your webcam. For this program to work properly all the automatic settings of your webcam have to be disabled (auto-focus, auto-brightness etc).

#### Software usage
After configuring your webcam you can close the window. You'll now have four black windows. To use the program perform this sequence of actions:
1. Without showing your hand in the frame, press the `B` key on your keyboard. It will start the process of background removal.
2. Move your hand so that it completely covers the two purple rectagles shown in the window called `output`. Press the `S` key on your keyboard. This will sample the color of your skin and start the process of hand detection.
3. When you want to close the program press the `esc` key on your keyboard.

#### Windows description
- The `output` window shows the output of the application. That is the RGB frame with the contour of the hand and a number corresponding to the number of lifted fingers. The purple number at the center of the hand represents the number of lifted fingers detected by the software.
- The `foreground` window shows the output of the background removal operation. From here you can easily see if background removal is performed correctly or not. If not, recalibrate by pressing `B`.
- The `handMask` window shows the output of the whole binarization process. Thats the image that is used for detecting the hand and counting its fingers.
- The `handDetection` window shows the output of the hand and finger detection process. The numbers shown here correspond to the index of the points in the arrays (read below). The purple number at the center of the hand represents the number of lifted fingers detected by the software.

## Hand segmentation
With segmentation we refer to the process of extracting objects of interest from an image. In our case, the object of interest is the hand of the user.

There are many possible approaches to solve this problem, each with different complexity and accuracy. Due to the lack of strong constraints on the scene's composition and illumination, we had to exclude segmentation techniques based on thresholding grayscale images. Even though this techniques are generally fast and reliable (with the right images), our typical image histogram doesn't have any recognizable separation of modes, therefore grayscale analysis is not a viable option.

We decided that the best approach to solve our problem, in terms of complexity and reliability, is to segment the hand starting from the color of the user's skin. The idea is quite simple: find the color of the user's skin and use it as a threshold to binarize the image.

### How to find the color of the user's skin
In order to find the skin color we decided to designate two specific areas of the screen as "sample areas". The user has to move his hand in the frame so that it covers the two sample areas. When the `S` key is pressed on the keyboard, the program saves the images contained in the sample areas, makes an average of the colors and then uses those two averaged colors as lower and higher thresholds to find the user's skin.

This solution is extremely simple and could be impoved in many ways. For example accuracy could be improved by taking more samples over time, to account for noise and light variations.

The code responsible for this part can be found [here](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/SkinDetector.cpp).

![sample areas image](https://raw.githubusercontent.com/PierfrancescoSoffritti/Handy/master/pictures/sample_areas.png)

#### Threshold calculation
The image is first converted to the HSV color space. After some testing this color space resulted the most resisted to changes of shade and tonality of the color, but is not the only option, YCrCb was also giving good results. The most significant advantage of using HSV is the ability to ignore shadows on the hand, simply by ignoring the third channel (V).

The sample images are divided in their three channels: H, S and V. For each channel we calculate the mean value, from there low and high threshold values are computed, except for the "value" channel (V).

After some testing we've found that lowering the low thresholds and increasing the high thresholds by a constant amount yields better results.

```
void SkinDetector::calculateThresholds(Mat sample1, Mat sample2) {
  int offsetLowThreshold = 80;
  int offsetHighThreshold = 30;

  Scalar hsvMeansSample1 = mean(sample1);
  Scalar hsvMeansSample2 = mean(sample2);

  hLowThreshold = min(hsvMeansSample1[0], hsvMeansSample2[0]) - offsetLowThreshold;
  hHighThreshold = max(hsvMeansSample1[0], hsvMeansSample2[0]) + offsetHighThreshold;

  sLowThreshold = min(hsvMeansSample1[1], hsvMeansSample2[1]) - offsetLowThreshold;
  sHighThreshold = max(hsvMeansSample1[1], hsvMeansSample2[1]) + offsetHighThreshold;

  vLowThreshold = 0;
  vHighThreshold = 255;
}
```

#### Binarization
The binarization of the frame is done using OpenCV's `inRange` function. The operator simply sets to 1 all the pixels contained between the low and high thresholds and to 0 all the other pixels.

`inRange(hsvInput, Scalar(hLowThreshold, sLowThreshold, vLowThreshold), Scalar(hHighThreshold, sHighThreshold, vHighThreshold), skinMask);`

After binarization the image resulted a bit noisy, because of false positives. To clean the image and remove the false positives, an opening operator is applied, with a 3x3 circular structuring element.

A dilation is also applied, just in case parts of the hand have been detached after binarization (sometimes fingers are detached from the hand).

```
performOpening(skinMask, MORPH_ELLIPSE, { 3, 3 });
dilate(skinMask, skinMask, Mat(), Point(-1, -1), 3);

void performOpening(Mat binaryImage, int kernelShape, Point kernelSize) {
  Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
  morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
}	
```

![final binary image](https://raw.githubusercontent.com/PierfrancescoSoffritti/Handy/master/pictures/binary_image.png)

### Remove the user's face
The skin of the user is now the object in our binary image.
Along with the user's hand, the user's face is also part of our object. This is obviously not desirable. 

To prevent our algorithm from picking up the face of the user, before the binarization of the image the user's face is detected and removed by drawing a black rectangle over it.
The problem of face detection is solved using one of the face classifiers provided by OpenCV.

The code responsible for this part can be found [here](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/FaceDetector.cpp).


### Background removal
At this point the program is working but, due to the unpredictability of the scene conditions, it is not particularly reliable. A simple change of illumination or a background with a color too similar to the color of the user's skin may give a lot of false positives.
In order to solve this problem we decided to add an extra step to our process: background removal, before binarization.

In a first approach we tried to use dynamic background subtraction. The problem with this solution is that the hand has to always move, otherwise it is classified as background and then removed.

Considering that this application is supposed to be used indoors, ideally at a desk, we can assume our image to be static (it doesn't change significantly over time). We decided to save a frame (converted to grayscale), and use it as a reference for background removal.

When the application starts the first frame is saved as reference. Then for each new frame we simply iterate over each pixel of the frame and compare it to the correspondig pixel in the reference frame. If the pixel in the current frame differs by a certain amount from the corresponding pixel in the reference frame it is not removed, otherwise the pixel is classified as background and removed.

More in detail:
* if `bg` is the reference frame and `input` is the current frame.
* If `bg[x,y] - offset <= input[x, y] <= bg[x, y] + offset`, input[x, y] is classified as background and set to 0. Every other pixel will be in the foreground.

```
void BackgroundRemover::removeBackground(Mat input, Mat background) {
  int thresholdOffset = 10;

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      uchar framePixel = input.at<uchar>(i, j);
      uchar bgPixel = background.at<uchar>(i, j);
    
      if (framePixel >= bgPixel - thresholdOffset && framePixel <= bgPixel + thresholdOffset)
        input.at<uchar>(i, j) = 0;
      else
        input.at<uchar>(i, j) = 255;
    }
  }
}
```

The code responsible for this part can be found [here](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/BackgroundRemover.cpp).

Background removal, applied before the binarization of the image, gave us good results and high accuracy in most scenes.

Despite working well this approach has many problems and limitation, if the background isn't static or if the illumination of the scene changes, it doesn't work.

To increase the flexibility of our program we assigned a key to the keyboard (the `B` key) that the user can press to replace the reference frame with the current frame. Even if the initial background changes (invalidating the original reference frame), the user can easily take a new sample.

![foreground](https://raw.githubusercontent.com/PierfrancescoSoffritti/Handy/master/pictures/foreground.png)

## Hand and finger detection

### Hand contour
Now that we have the binary image, we use OpenCV's function `findContours` to get the contours of all objects in the image. From these we select the contour with the biggest area. If the binarization of the image has been done correctly, this contour should be the one of our hand.

At this point we look for the smallest convex set containing the hand contour, using the `convexHull` function. We then construct the bounding rectangle of the convex hull. The rectangle will be used to calculate an approximation of the center of the hand and to do scale invariant computations.

The code responsible for this part can be found [here](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/FingerCount.cpp).

### Fingers identification
The points of intersection between the hand contour and convex hull are saved in an array, they will be used to locate the finger tips.

Using the `convexityDefects` function, we get all the defects of the contour and we save them in another array. These are the lowest points between one finger and the other. We can use the two arrays to make assumptions about the number of lifted fingers in the image.

`convexityDefects` usually returns more points than we need, therefore we have to filter them. We filter them based on their distance from the center of the bounding rectangle (which approximately corresponds to the center of the hand), so that only the lowest points between each finger are kept. In order to make this process scale invariant, we use the height of the bounding rectangle as reference.

Both arrays (farthest points from the convex hull and closest points to the convex hull) are then filtered again, we need exactly one point for each finger tip and one point between each finger. To do that all points are averaged using a chosen neighborhood: for each point, all the points within a certain distance are averaged to a single point.

Now we can analyze our arrays in order to detect fingers.

For each point in the finger tips array we look in the defects array for the two nearest points on the x axis. We now have three points: one is a fingertip candidate (we aren't sure yet, the others are "local minimum" representing the concavities of the hand.

![finger tips and convex hull defects](https://raw.githubusercontent.com/PierfrancescoSoffritti/Handy/master/pictures/tips_and_defects.png)

To determine if the point is really a fingertip, the following steps are performed:
1. Check that the angle between the three points is within specified limits (usually the angle between the tip of our finger and the two closest concavities is within a certain range).
2. Check that the y coordinate of the finger tip is not lower than the y coordinates of the two concavity points (our hand is supposed not to be upside down).
3. Check that the y coordinates of the two concavity points are not lower than the y coordinate of the center of the hand (in the case of the thumb and pinkie fingers one point is allowed to be below the center, but both would be anatomically incorrect).
4. Check that the distance between the center of the hand and the finger tip is greater than a chosen limit, scaled with the height of the bounding rectangle (fingers shouldn't be too small or big compared to the size of the hand).
5. To increase accuracy for the case in which we have no lifted finger, we check that all the concavity points have a minimum distance from the center of the hand. This minimum distance is scaled using the height of the bounding rectangle.

To remove false positives, every point following the fifth fingertip is removed.

The code responsible for this part can be found [here](https://github.com/PierfrancescoSoffritti/Handy/blob/master/Handy/FingerCount.cpp).

![hand contour](https://raw.githubusercontent.com/PierfrancescoSoffritti/Handy/master/pictures/contour.png)

## Limitations and possible improvements
* The accuracy of this solution is not the always the best and the program may need to be tuned differently for different environments.
* This sofware has been written with limited knowledge of both OpenCV and C++, there's probably plenty of room for optimizations and performance improvements.
* As mentioned before, the samples of the user's skins are collocted in an exremely simplistic way. There's a lot of room for improvements here.
* Some part of the application rely on a few "magic numbers" (hardcoded constants), it would be ideal to find a way to get rid of them.
