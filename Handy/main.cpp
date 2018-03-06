#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include"SkinDetector.h"
#include"FaceDetector.h"

using namespace cv;
using namespace std;

int main(int, char**) {
	String windowName = "videoCapture";

	VideoCapture videoCapture(0);
	
	if (!videoCapture.isOpened())
		return -1;

	namedWindow(windowName, 1);
	Mat frame;

	SkinDetector skinDetector;
	FaceDetector faceDetector;

	while (true) {
		videoCapture >> frame;

		//frame = skinDetector.detectSkin(frame);
		faceDetector.detectFaces(frame, frame);

		imshow(windowName, frame);
		
		if (waitKey(1) >= 0) break;
	}

	return 0;
}