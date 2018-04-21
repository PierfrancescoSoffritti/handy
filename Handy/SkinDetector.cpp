#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

/*
 Author: Pierfrancesco Soffritti
*/

SkinDetector::SkinDetector(void) {
	hLowerThreshold = 0;
	hHigherThreshold = 0;
	sLowerThreshold = 0;
	sHigherThreshold = 0;
	vLowerThreshold = 0;
	vHigherThreshold = 0;

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

void SkinDetector::calculateThresholds(Mat sample1, Mat sample2) {
	int offsetMinThreshold = 80;
	int offsetMaxThreshold = 30;

	Scalar hsv_means_sample1 = mean(sample1);
	Scalar hsv_means_sample2 = mean(sample2);

	hLowerThreshold = min(hsv_means_sample1[0], hsv_means_sample2[0]) - offsetMinThreshold;
	hHigherThreshold = max(hsv_means_sample1[0], hsv_means_sample2[0]) + offsetMaxThreshold;

	sLowerThreshold = min(hsv_means_sample1[1], hsv_means_sample2[1]) - offsetMinThreshold;
	sHigherThreshold = max(hsv_means_sample1[1], hsv_means_sample2[1]) + offsetMaxThreshold;

	//vLowerThreshold = min(hsv_means_sample1[2], hsv_means_sample2[2]) - offsetMinThreshold;
	//vHigherThreshold = max(hsv_means_sample1[2], hsv_means_sample2[2]) + offsetMaxThreshold;
	vLowerThreshold = 0;
	vHigherThreshold = 255;
}

Mat SkinDetector::getSkinMask(Mat input) {
	Mat skinMask;

	if (!calibrated) {
		skinMask = Mat::zeros(input.size(), CV_8UC1);
		return skinMask;
	}

	Mat hsvInput;
	cvtColor(input, hsvInput, CV_BGR2HSV);

	inRange(
		hsvInput,
		Scalar(hLowerThreshold, sLowerThreshold, vLowerThreshold),
		Scalar(hHigherThreshold, sHigherThreshold, vHigherThreshold),
		skinMask);

	performOpening(skinMask, MORPH_ELLIPSE, { 3, 3 }, 2);

	return skinMask;
}

void SkinDetector::performOpening(Mat binaryImage, int kernelShape, Point kernelSize, int interations) {
	Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
	morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
}