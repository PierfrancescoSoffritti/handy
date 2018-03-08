#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void) {
	//YCrCb threshold
	Y_MIN = 0;
	Y_MAX = 255;
	Cr_MIN = 133;
	Cr_MAX = 173;
	Cb_MIN = 77;
	Cb_MAX = 127;
}

Rect r1, r2;

void SkinDetector::drawRect(Mat input) {
	int with = input.size().width, height = input.size().height;

	r1 = Rect(with / 2, height / 2, 20, 20);
	r2 = Rect(with / 2, height / 3, 20, 20);

	rectangle(
		input,
		r1,
		Scalar(255, 0, 255)
	);

	rectangle(
		input,
		r2,
		Scalar(255, 0, 255)
	);
}

void SkinDetector::calibrate(Mat input) {

	//Mat skin;
	//cvtColor(input, skin, COLOR_BGR2YCrCb);

	Mat m1 = Mat(input, r1);
	Mat m2 = Mat(input, r2);

	vector<Mat> channels1;
	vector<Mat> channels2;

	split(m1, channels1);
	split(m2, channels2);

	int offsetMin = 60;
	int offsetMax = 30;

	Y_MIN = min( mean(channels1[0])[0], mean(channels2[0])[0]) - offsetMin;
	Y_MAX = max( mean(channels1[0])[0], mean(channels2[0])[0]) + offsetMax;

	Cr_MIN = min( mean(channels1[1])[0], mean(channels2[1])[0]) - offsetMin;
	Cr_MAX = max( mean(channels1[1])[0], mean(channels2[1])[0]) + offsetMax;

	Cb_MIN = min( mean(channels1[2])[0], mean(channels2[2])[0]) - offsetMin;
	Cb_MAX = max( mean(channels1[2])[0], mean(channels2[2])[0]) + offsetMax;
}

Mat SkinDetector::detectSkin(Mat input) {
	if (Y_MIN == 0)
		return input;

	//Mat skin;
	//cvtColor(input, input, COLOR_BGR2YCrCb);

	//filter the image in YCrCb color space
	inRange(input, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), input);

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, { 11, 11 });
	erode(input, input, kernel, Point(-1,-1), 1);
	dilate(input, input, kernel, Point(-1, -1), 1);

	GaussianBlur(input, input, { 3, 3 }, 0);
	bitwise_and(input, input, input);

	return input;
}