#pragma once
#include<opencv\cv.h>

class FingerCount {
public:
	FingerCount(void);
	cv::Mat findHandContours(cv::Mat input);
};