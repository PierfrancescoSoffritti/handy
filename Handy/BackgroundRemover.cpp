#include "BackgroundRemover.h"
#include"opencv2\opencv.hpp"

BackgroundRemover::BackgroundRemover(void) {
	background;
	calibrated = false;
}

void BackgroundRemover::calibrate(Mat input) {
	cvtColor(input, background, CV_BGR2GRAY);
	calibrated = true;
}

Mat BackgroundRemover::getForeground(Mat input) {
	Mat foregroundMask = getForegroundMask(input);

	//imshow("foregroundMask", foregroundMask);

	Mat foreground;
	input.copyTo(foreground, foregroundMask);

	return foreground;
}

Mat BackgroundRemover::getForegroundMask(Mat input) {
	Mat foregroundMask;

	if (!calibrated) {
		foregroundMask = Mat::zeros(input.size(), CV_8UC1);
		return foregroundMask;
	}

	cvtColor(input, foregroundMask, CV_BGR2GRAY);

	int offset = 10;

	for (int i = 0; i < foregroundMask.rows; i++) {
		for (int j = 0; j < foregroundMask.cols; j++) {
			uchar framePixel = foregroundMask.at<uchar>(i, j);
			uchar bgPixel = background.at<uchar>(i, j);

			if (framePixel > bgPixel - offset && framePixel < bgPixel + offset)
				foregroundMask.at<uchar>(i, j) = 0;
			else
				foregroundMask.at<uchar>(i, j) = 255;
		}
	}

	//imshow("mask", mask);
	
	return foregroundMask;
}