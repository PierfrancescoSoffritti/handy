#include "FaceDetector.h"
#include"opencv2\opencv.hpp"

using namespace cv;
using namespace std;

String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

FaceDetector::FaceDetector(void) {
	if (!face_cascade.load(face_cascade_name))
		throw std::runtime_error("can't load file " + face_cascade_name);
}

void FaceDetector::detectFaces(cv::Mat input, cv::Mat output) {
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(input, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(output,
			Point(faces[i].x, faces[i].y),
			Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
			Scalar(255, 0, 255)
		);
	}
}