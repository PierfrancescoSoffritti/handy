#pragma once

#include<opencv\cv.h>

class FaceDetector {
public:
	FaceDetector(void);

	void detectFaces(cv::Mat input, cv::Mat output);
	cv::Mat getSkinHistogram(cv::Mat input);
};