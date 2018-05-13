#include "FaceDetector.h"
#include"opencv2\opencv.hpp"

/*
 Author: Pierfrancesco Soffritti https://github.com/PierfrancescoSoffritti
*/

Rect getFaceRect(Mat input);

String faceClassifierFileName = "../res/haarcascade_frontalface_alt.xml";
CascadeClassifier faceCascadeClassifier;

FaceDetector::FaceDetector(void) {
	if (!faceCascadeClassifier.load(faceClassifierFileName))
		throw runtime_error("can't load file " + faceClassifierFileName);
}

void FaceDetector::removeFaces(Mat input, Mat output) {
	vector<Rect> faces;
	Mat frameGray;

	cvtColor(input, frameGray, CV_BGR2GRAY);
	equalizeHist(frameGray, frameGray);

	faceCascadeClassifier.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

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

	faceCascadeClassifier.detectMultiScale(inputGray, faceRectangles, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

	if (faceRectangles.size() > 0)
		return faceRectangles[0];
	else
		return Rect(0, 0, 1, 1);
}