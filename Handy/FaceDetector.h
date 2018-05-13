#pragma once

#include<opencv\cv.h>

/*
 Author: Pierfrancesco Soffritti https://github.com/PierfrancescoSoffritti
*/

using namespace cv;
using namespace std;

class FaceDetector {
	public:
		FaceDetector(void);
		void removeFaces(Mat input, Mat output);
};