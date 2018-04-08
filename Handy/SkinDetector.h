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
		int h_min = 0;
		int h_max = 0;
		int s_min = 0;
		int s_max = 0;
		int v_min = 0;
		int v_max = 0;

		bool calibrated = false;

		Rect skinColorSamplerRectangle1, skinColorSamplerRectangle2;

		void calculateThresholds(vector<Mat> hsvChannelsSample1, vector<Mat> hsvChannelsSample2);
};