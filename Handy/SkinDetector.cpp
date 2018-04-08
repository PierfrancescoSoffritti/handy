#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void) {
	h_min = 0;
	h_max = 0;
	s_min = 0;
	s_max = 0;
	v_min = 0;
	v_max = 0;

	calibrated = false;

	skinColorSamplerRectangle1, skinColorSamplerRectangle2;
}

void SkinDetector::drawSkinColorSampler(Mat input) {
	int frameWidth = input.size().width, frameHeight = input.size().height;

	int rectangleSize = 20;
	Scalar rectangleColor = Scalar(255, 0, 255);

	skinColorSamplerRectangle1 = Rect(frameWidth / 5, frameHeight / 2, rectangleSize, rectangleSize);
	skinColorSamplerRectangle2 = Rect(frameWidth / 5, frameHeight / 3, rectangleSize, rectangleSize);

	rectangle(
		input,
		skinColorSamplerRectangle1,
		rectangleColor
	);

	rectangle(
		input,
		skinColorSamplerRectangle2,
		rectangleColor
	);
}

void SkinDetector::calibrate(Mat input) {
	
	Mat hsvInput;
	cvtColor(input, hsvInput, CV_BGR2HSV);

	Mat sample1 = Mat(hsvInput, skinColorSamplerRectangle1);
	Mat sample2 = Mat(hsvInput, skinColorSamplerRectangle2);

	calculateThresholds(sample1, sample2);

	calibrated = true;
}

void SkinDetector::calculateThresholds(Mat sample1,Mat sample2) {
	int offsetMinThreshold = 80;
	int offsetMaxThreshold = 30;

	Scalar hsv_means_sample1 = mean(sample1);
	Scalar hsv_means_sample2 = mean(sample2);

	h_min = min(hsv_means_sample1[0], hsv_means_sample2[0]) - offsetMinThreshold;
	h_max = max(hsv_means_sample1[0], hsv_means_sample2[0]) + offsetMaxThreshold;

	s_min = min(hsv_means_sample1[1], hsv_means_sample2[1]) - offsetMinThreshold;
	s_max = max(hsv_means_sample1[1], hsv_means_sample2[1]) + offsetMaxThreshold;

	//v_min = min(hsv_means_sample1[2], hsv_means_sample2[2]) - offsetMinThreshold;
	//v_max = max(hsv_means_sample1[2], hsv_means_sample2[2]) + offsetMaxThreshold;
	v_min = 0;
	v_max = 255;
}

Mat SkinDetector::getSkinMask(Mat input) {
	Mat skinMask;

	if (!calibrated) {
		skinMask = Mat::zeros(input.size(), CV_8UC1);
		return skinMask;
	}

	Mat hsvInput;
	cvtColor(input, hsvInput, CV_BGR2HSV);

	inRange(hsvInput, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), skinMask);

	//Mat kernel = getStructuringElement(MORPH_ELLIPSE, { 11, 11 });
	//erode(skinMask, skinMask, kernel, Point(-1,-1), 1);
	//dilate(skinMask, skinMask, kernel, Point(-1, -1), 1);

	GaussianBlur(skinMask, skinMask, { 3, 3 }, 0);
	bitwise_and(skinMask, skinMask, skinMask);

	return skinMask;
}