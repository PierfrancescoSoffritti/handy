#include "FaceDetector.h"
#include"opencv2\opencv.hpp"

/*
Author: Pierfrancesco Soffritti
*/

Rect getFaceRect(Mat input);

String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

FaceDetector::FaceDetector(void) {
	if (!face_cascade.load(face_cascade_name))
		throw runtime_error("can't load file " + face_cascade_name);
}

void FaceDetector::removeFaces(Mat input, Mat output) {
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(input, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(
			output,
			Point(faces[i].x, faces[i].y),
			Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
			Scalar(0, 0, 0),
			-1
		);
	}
}

Rect getFaceRect(Mat input) {
	vector<Rect> faceRectangles;
	Mat inputGray;

	cvtColor(input, inputGray, CV_BGR2GRAY);
	equalizeHist(inputGray, inputGray);

	face_cascade.detectMultiScale(inputGray, faceRectangles, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

	if (faceRectangles.size() > 0)
		return faceRectangles[0];
	else
		return Rect(0, 0, 1, 1);
}