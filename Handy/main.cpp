#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include "SkinDetector.h"
#include "FaceDetector.h"
#include "FingerCount.h"

using namespace cv;
using namespace std;

Mat getHistogram(Mat in, Mat out, int bins);
Mat getBackProjection(Mat in, Mat histogram);

int main(int, char**) {
	VideoCapture videoCapture(0);
	
	if (!videoCapture.isOpened()) {
		cout << "Can't find camera!" << endl;
		return -1;
	}

	Mat frame;
	Mat skinMask;
	Mat contourImage;

	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;

	while (true) {
		videoCapture >> frame;

		cvtColor(frame, frame, CV_BGR2HSV);

		Mat kernel = getStructuringElement(MORPH_ELLIPSE, { 4, 4 });
		erode(frame, frame, kernel, Point(-1, -1), 3);
		dilate(frame, frame, kernel, Point(-1, -1), 2);
		
		faceDetector.removeFaces(frame, frame);
		skinDetector.drawRect(frame);
		skinMask = skinDetector.detectSkin(frame);
		contourImage = fingerCount.findHandContours(skinMask);
		
		/*
		Mat histogram;
		histogram = getHistogram(frame, histogram, 25);

		Mat skinHistogram = faceDetector.getSkinHistogram(frame);
		frame = getBackProjection(frame, skinHistogram);	
		*/

		imshow("frame", frame);
		imshow("skinMask", skinMask);
		imshow("hand contour", contourImage);

		int key = waitKey(1);

		if (key == 27)
			break;
		else if(key > 0)
			skinDetector.calibrate(frame);
	}

	return 0;
}

Mat getBackProjection(Mat input, Mat histogram) {
	
	int numImages = 1;
	int channels[] = { 0 };

	float range[] = { 120, 200 };
	const float* ranges[] = { range };

	Mat backProjection;
	calcBackProject(&input, numImages, channels, histogram, backProjection, ranges);

	return backProjection;
}

Mat getHistogram(Mat in, Mat out, int bins) {
	cvtColor(in, in, CV_BGR2HSV);

	// Use only the Hue value
	out.create(in.size(), in.type());
	// channel 0 of in goes to channel 0 of out
	int ch[] = { 0,0, 1,1, 2,2 };
	mixChannels(&in, 1, &out, 1, ch, 3);

	// ---
	Mat histogram;
	int histogramSize = bins;
	float hue_range[] = { 0, 180 };
	float saturation_range[] = { 0, 180 };
	float value_range[] = { 0, 180 };
	const float* ranges[] = { hue_range, saturation_range, value_range };

	// Get the Histogram and normalize it
	calcHist(&out, 1, ch, Mat(), histogram, 1, &histogramSize, ranges, true, false);
	normalize(histogram, histogram, 0, 255, NORM_MINMAX, -1, Mat());

	// Draw the histogram
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histogramSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	for (int i = 0; i < bins; i++)
		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(histogram.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);

	imshow("hist", histImg);

	// Get Backprojection
	Mat backproj;
	calcBackProject(&out, 1, ch, histogram, backproj, ranges, 1, true);

	// Draw the backproj
	imshow("BackProj", backproj);

	// ---

	return out;
}