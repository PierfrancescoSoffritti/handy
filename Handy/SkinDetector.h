#pragma once

#include<opencv\cv.h>

using namespace cv;
using namespace std;

class SkinDetector {
	public:
		SkinDetector(void);
		bool calibrated;

		void drawSampleRects(Mat input);
		void calibrate(Mat input);
		Mat detectSkin(Mat input);		

	private:
		int H_MIN;
		int H_MAX;
		int S_MIN;
		int S_MAX;
		int V_MIN;
		int V_MAX;
};