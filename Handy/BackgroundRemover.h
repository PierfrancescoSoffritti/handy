#pragma once

#include<opencv\cv.h>

using namespace cv;
using namespace std;

class BackgroundRemover {
	public:
		BackgroundRemover(void);
		void calibrate(Mat input);
		Mat BackgroundRemover::getForeground(Mat input);

	private:
		Mat background;
		bool calibrated = false;

		Mat getForegroundMask(Mat input);
};