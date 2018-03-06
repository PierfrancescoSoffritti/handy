#include "SkinDetector.h"
#include"opencv2\opencv.hpp"

SkinDetector::SkinDetector(void) {
	//YCrCb threshold
	Y_MIN = 150;
	Y_MAX = 255;
	Cr_MIN = 133;
	Cr_MAX = 173;
	Cb_MIN = 77;
	Cb_MAX = 127;
}

cv::Mat SkinDetector::detectSkin(cv::Mat input) {
	cv::Mat skin;

	cv::cvtColor(input, skin, cv::COLOR_BGR2YCrCb);

	//filter the image in YCrCb color space
	cv::inRange(skin, cv::Scalar(Y_MIN, Cr_MIN, Cb_MIN), cv::Scalar(Y_MAX, Cr_MAX, Cb_MAX), skin);

	return skin;
}