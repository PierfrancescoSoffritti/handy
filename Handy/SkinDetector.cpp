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

	vector<Mat> hsvChannelsSample1;
	vector<Mat> hsvChannelsSample2;

	split(sample1, hsvChannelsSample1);
	split(sample2, hsvChannelsSample2);

	calculateThresholds(hsvChannelsSample1, hsvChannelsSample2);

	calibrated = true;
}

void SkinDetector::calculateThresholds(vector<Mat> hsvChannelsSample1, vector<Mat> hsvChannelsSample2) {
	int offsetMinThreshold = 80;
	int offsetMaxThreshold = 30;

	h_min = min(mean(hsvChannelsSample1[0])[0], mean(hsvChannelsSample2[0])[0]) - offsetMinThreshold;
	h_max = max(mean(hsvChannelsSample1[0])[0], mean(hsvChannelsSample2[0])[0]) + offsetMaxThreshold;

	s_min = min(mean(hsvChannelsSample1[1])[0], mean(hsvChannelsSample2[1])[0]) - offsetMinThreshold;
	s_max = max(mean(hsvChannelsSample1[1])[0], mean(hsvChannelsSample2[1])[0]) + offsetMaxThreshold;

	v_min = min(mean(hsvChannelsSample1[2])[0], mean(hsvChannelsSample2[2])[0]) - offsetMinThreshold;
	v_max = max(mean(hsvChannelsSample1[2])[0], mean(hsvChannelsSample2[2])[0]) + offsetMaxThreshold;
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