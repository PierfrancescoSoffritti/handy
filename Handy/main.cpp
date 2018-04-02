#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include "BackgroundRemover.h"
#include "SkinDetector.h"
#include "FaceDetector.h"
#include "FingerCount.h"

using namespace cv;
using namespace std;

int main(int, char**) {
	VideoCapture videoCapture(0);
	videoCapture.set(CV_CAP_PROP_SETTINGS, 1);

	if (!videoCapture.isOpened()) {
		cout << "Can't find camera!" << endl;
		return -1;
	}

	Mat frame, handMask, foreground, fingerCountDebug;

	BackgroundRemover backgroundRemover;
	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;

	while (true) {
		videoCapture >> frame;

		skinDetector.drawSkinColorSampler(frame);

		foreground = backgroundRemover.getForeground(frame);
		
		faceDetector.removeFaces(frame, foreground);
		handMask = skinDetector.getSkinMask(foreground);
		fingerCountDebug = fingerCount.findFingersCount(handMask, frame);

		imshow("output", frame);
		imshow("foreground", foreground);
		imshow("handMask", handMask);
		imshow("fingerCountDebug", fingerCountDebug);
		
		int key = waitKey(1);

		if (key == 27) // esc
			break;
		else if (key == 98) // b
			backgroundRemover.calibrate(frame);
		else if (key == 115) // s
			skinDetector.calibrate(frame);
	}

	return 0;
}