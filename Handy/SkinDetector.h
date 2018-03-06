#pragma once

#include<opencv\cv.h>

class SkinDetector {
	public:
		SkinDetector(void);

		cv::Mat detectSkin(cv::Mat input);

	private:
		int Y_MIN;
		int Y_MAX;
		int Cr_MIN;
		int Cr_MAX;
		int Cb_MIN;
		int Cb_MAX;
};