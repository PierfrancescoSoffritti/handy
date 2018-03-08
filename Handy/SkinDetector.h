#pragma once

#include<opencv\cv.h>

using namespace cv;
using namespace std;

class SkinDetector {
	public:
		SkinDetector(void);

		void drawRect(Mat input);
		void calibrate(Mat input);
		Mat detectSkin(Mat input);		

	private:
		int Y_MIN;
		int Y_MAX;
		int Cr_MIN;
		int Cr_MAX;
		int Cb_MIN;
		int Cb_MAX;
};