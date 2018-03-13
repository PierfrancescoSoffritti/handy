#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void) {
	//HSV threshold
	H_MIN = 0;
	H_MAX = 255;
	S_MIN = 133;
	S_MAX = 173;
	V_MIN = 77;
	V_MAX = 127;

	calibrated = false;
}

Rect sampleRect1, sampleRect2;

void SkinDetector::drawSampleRects(Mat input) {
	int width = input.size().width, height = input.size().height;

	sampleRect1 = Rect(width / 4, height / 2, 20, 20);
	sampleRect2 = Rect(width / 4, height / 3, 20, 20);

	rectangle(
		input,
		sampleRect1,
		Scalar(255, 0, 255)
	);

	rectangle(
		input,
		sampleRect2,
		Scalar(255, 0, 255)
	);
}

void SkinDetector::calibrate(Mat input) {

	Mat sample1 = Mat(input, sampleRect1);
	Mat sample2 = Mat(input, sampleRect2);

	vector<Mat> channelsSample1;
	vector<Mat> channelsSample2;

	split(sample1, channelsSample1);
	split(sample2, channelsSample2);

	int offsetMin = 60;
	int offsetMax = 30;

	H_MIN = min( mean(channelsSample1[0])[0], mean(channelsSample2[0])[0]) - offsetMin;
	H_MAX = max( mean(channelsSample1[0])[0], mean(channelsSample2[0])[0]) + offsetMax;

	S_MIN = min( mean(channelsSample1[1])[0], mean(channelsSample2[1])[0]) - offsetMin;
	S_MAX = max( mean(channelsSample1[1])[0], mean(channelsSample2[1])[0]) + offsetMax;

	V_MIN = min( mean(channelsSample1[2])[0], mean(channelsSample2[2])[0]) - offsetMin;
	V_MAX = max( mean(channelsSample1[2])[0], mean(channelsSample2[2])[0]) + offsetMax;

	calibrated = true;
}

Mat SkinDetector::detectSkin(Mat input) {
	if (!calibrated)
		return input;

	Mat skin;
	inRange(input, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), skin);

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, { 11, 11 });
	//erode(skin, skin, kernel, Point(-1,-1), 1);
	dilate(skin, skin, kernel, Point(-1, -1), 1);

	//GaussianBlur(skin, skin, { 3, 3 }, 0);
	//bitwise_and(skin, skin, skin);

	return skin;
}