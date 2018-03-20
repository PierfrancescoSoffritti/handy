#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void) {
	H_MIN = 0;
	H_MAX = 0;
	S_MIN = 0;
	S_MAX = 0;
	V_MIN = 0;
	V_MAX = 0;

	calibrated = false;

	skinColorSamplerRectangle1, skinColorSamplerRectangle2;
}

void SkinDetector::drawSkinColorSampler(Mat frame) {
	int frameWidth = frame.size().width, frameHeight = frame.size().height;

	int rectangleSize = 20;
	Scalar rectangleColor = Scalar(255, 0, 255);

	skinColorSamplerRectangle1 = Rect(frameWidth / 5, frameHeight / 2, rectangleSize, rectangleSize);
	skinColorSamplerRectangle2 = Rect(frameWidth / 5, frameHeight / 3, rectangleSize, rectangleSize);

	rectangle(
		frame,
		skinColorSamplerRectangle1,
		rectangleColor
	);

	rectangle(
		frame,
		skinColorSamplerRectangle2,
		rectangleColor
	);
}

void SkinDetector::calibrate(Mat input) {
	
	Mat hsvInput;
	cvtColor(input, hsvInput, CV_BGR2HSV);

	Mat sample1 = Mat(hsvInput, skinColorSamplerRectangle1);
	Mat sample2 = Mat(hsvInput, skinColorSamplerRectangle2);

	vector<Mat> channelsSample1;
	vector<Mat> channelsSample2;

	split(sample1, channelsSample1);
	split(sample2, channelsSample2);

	int offsetMin = 80;
	int offsetMax = 30;

	H_MIN = min( mean(channelsSample1[0])[0], mean(channelsSample2[0])[0]) - offsetMin;
	H_MAX = max( mean(channelsSample1[0])[0], mean(channelsSample2[0])[0]) + offsetMax;

	S_MIN = min( mean(channelsSample1[1])[0], mean(channelsSample2[1])[0]) - offsetMin;
	S_MAX = max( mean(channelsSample1[1])[0], mean(channelsSample2[1])[0]) + offsetMax;

	V_MIN = min( mean(channelsSample1[2])[0], mean(channelsSample2[2])[0]) - offsetMin;
	V_MAX = max( mean(channelsSample1[2])[0], mean(channelsSample2[2])[0]) + offsetMax;

	calibrated = true;
}

Mat SkinDetector::getSkinMask(Mat input) {
	Mat skinMask;

	if (!calibrated) {
		skinMask = Mat::zeros(input.size(), CV_8UC1);
		return skinMask;
	}

	Mat hsvInput;
	cvtColor(input, hsvInput, CV_BGR2HSV);

	inRange(hsvInput, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), skinMask);

	//Mat kernel = getStructuringElement(MORPH_ELLIPSE, { 11, 11 });
	//erode(skinMask, skinMask, kernel, Point(-1,-1), 1);
	//dilate(skinMask, skinMask, kernel, Point(-1, -1), 1);

	GaussianBlur(skinMask, skinMask, { 3, 3 }, 0);
	bitwise_and(skinMask, skinMask, skinMask);

	return skinMask;
}