#pragma once

#include<opencv\cv.h>

using namespace cv;
using namespace std;

class SkinDetector {
	public:
		SkinDetector(void);

		void drawSkinColorSampler(Mat input);
		void calibrate(Mat input);
		Mat getSkinMask(Mat input);

	private:
		int H_MIN = 0;
		int H_MAX = 0;
		int S_MIN = 0;
		int S_MAX = 0;
		int V_MIN = 0;
		int V_MAX = 0;

		bool calibrated = false;

		Rect skinColorSamplerRectangle1, skinColorSamplerRectangle2;
};