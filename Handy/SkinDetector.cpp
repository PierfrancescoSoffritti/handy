#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

/*
 Author: Pierfrancesco Soffritti https://github.com/PierfrancescoSoffritti
*/

SkinDetector::SkinDetector(void) {
	hLowThreshold = 0;
	hHighThreshold = 0;
	sLowThreshold = 0;
	sHighThreshold = 0;
	vLowThreshold = 0;
	vHighThreshold = 0;

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
	int offsetLowThreshold = 80;
	int offsetHighThreshold = 30;

	Scalar hsvMeansSample1 = mean(sample1);
	Scalar hsvMeansSample2 = mean(sample2);

	hLowThreshold = min(hsvMeansSample1[0], hsvMeansSample2[0]) - offsetLowThreshold;
	hHighThreshold = max(hsvMeansSample1[0], hsvMeansSample2[0]) + offsetHighThreshold;

	sLowThreshold = min(hsvMeansSample1[1], hsvMeansSample2[1]) - offsetLowThreshold;
	sHighThreshold = max(hsvMeansSample1[1], hsvMeansSample2[1]) + offsetHighThreshold;

	// the V channel shouldn't be used. By ignorint it, shadows on the hand wouldn't interfire with segmentation.
	// Unfortunately there's a bug somewhere and not using the V channel causes some problem. This shouldn't be too hard to fix.
	vLowThreshold = min(hsvMeansSample1[2], hsvMeansSample2[2]) - offsetLowThreshold;
	vHighThreshold = max(hsvMeansSample1[2], hsvMeansSample2[2]) + offsetHighThreshold;
	//vLowThreshold = 0;
	//vHighThreshold = 255;
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
		Scalar(hLowThreshold, sLowThreshold, vLowThreshold),
		Scalar(hHighThreshold, sHighThreshold, vHighThreshold),
		skinMask);

	performOpening(skinMask, MORPH_ELLIPSE, { 3, 3 });
	dilate(skinMask, skinMask, Mat(), Point(-1, -1), 3);

	return skinMask;
}

void SkinDetector::performOpening(Mat binaryImage, int kernelShape, Point kernelSize) {
	Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
	morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
}