#pragma once

#include<opencv\cv.h>

using namespace cv;
using namespace std;

class BackgroundRemover {
	public:
		BackgroundRemover(void);
		void calibrate(Mat input);
		Mat BackgroundRemover::getForegroundMask(Mat input);
};