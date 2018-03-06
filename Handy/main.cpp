#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**) {
	String windowName = "videoCapture";

	VideoCapture videoCapture(0);
	
	if (!videoCapture.isOpened())
		return -1;

	namedWindow(windowName, 1);
	Mat frame;

	while (true) {
		videoCapture >> frame;
		imshow(windowName, frame);
		
		if (waitKey(1) >= 0) break;
	}

	return 0;
}