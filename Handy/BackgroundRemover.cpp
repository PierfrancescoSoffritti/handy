#include "BackgroundRemover.h"
#include"opencv2\opencv.hpp"

Mat background;
bool calibrated = false;

BackgroundRemover::BackgroundRemover(void) {
}

void BackgroundRemover::calibrate(Mat input) {
	cvtColor(input, background, CV_BGR2GRAY);
	calibrated = true;
}

Mat BackgroundRemover::getForegroundMask(Mat input) {
	Mat mask;

	if (!calibrated) {
		mask = Mat::zeros(Size(input.cols, input.rows), CV_8UC1);
		return mask;
	}

	cvtColor(input, mask, CV_BGR2GRAY);

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			uchar framePixel = mask.at<uchar>(i, j);
			uchar bgPixel = background.at<uchar>(i, j);

			if (framePixel > bgPixel - 10 && framePixel < bgPixel + 10)
				mask.at<uchar>(i, j) = 0;
		}
	}

	imshow("mask", mask);
	
	return mask;
}