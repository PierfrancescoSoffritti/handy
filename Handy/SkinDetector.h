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
		int hLowerThreshold = 0;
		int hHigherThreshold = 0;
		int sLowerThreshold = 0;
		int sHigherThreshold = 0;
		int vLowerThreshold = 0;
		int vHigherThreshold = 0;

		bool calibrated = false;

		Rect skinColorSamplerRectangle1, skinColorSamplerRectangle2;

		void calculateThresholds(Mat sample1, Mat sample2);
		void SkinDetector::performOpening(Mat binaryImage, int structuralElementShapde, Point structuralElementSize, int interations);
};