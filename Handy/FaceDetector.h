#pragma once

#include<opencv\cv.h>

class FaceDetector {
	public:
		FaceDetector(void);
		void removeFaces(cv::Mat input, cv::Mat output);
};