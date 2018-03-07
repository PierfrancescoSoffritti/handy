#include "FaceDetector.h"
#include"opencv2\opencv.hpp"

using namespace cv;
using namespace std;

String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

FaceDetector::FaceDetector(void) {
	if (!face_cascade.load(face_cascade_name))
		throw runtime_error("can't load file " + face_cascade_name);
}

void FaceDetector::detectFaces(Mat input, Mat output) {
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

Rect getFaceRect(Mat input) {
	vector<Rect> faceRects;
	Mat frame_gray;

	cvtColor(input, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faceRects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(120, 120));

	if (faceRects.size() > 0) {
		faceRects[0].x += 15;
		faceRects[0].y += 15;

		faceRects[0].width /= 1.3;
		faceRects[0].height /= 1.3;

		return faceRects[0];
	} else
		return Rect(0, 0, 10, 10);
}

Mat getSkinInstogramFromFace(Mat input, int bins) {
	cvtColor(input, input, COLOR_BGR2YCrCb);

	int ch[] = { 1,0 };

	// ---
	Mat histogram;
	int histogramSize = bins;
	float hue_range[] = { 100, 200 };
	float saturation_range[] = { 0, 180 };
	const float* ranges[] = { hue_range, saturation_range };

	// Get the Histogram and normalize it
	calcHist(&input, 1, ch, Mat(), histogram, 1, &histogramSize, ranges, true, false);
	normalize(histogram, histogram, 0, 255, NORM_MINMAX, -1, Mat());

	/*
	// Draw the histogram
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histogramSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	for (int i = 0; i < bins; i++)
		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(histogram.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);

	imshow("hist", histImg);
	*/

	return histogram;
}

Mat FaceDetector::getSkinHistogram(Mat input) {
	Rect faceRect = getFaceRect(input);

	Mat face = Mat(input, faceRect);
	imshow("face", face);

	Mat skinHistogram = getSkinInstogramFromFace(face, 25);
	
	rectangle(
		input,
		Point(faceRect.x, faceRect.y),
		Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height),
		Scalar(0, 0, 0),
		-1
	);
	
	return skinHistogram;
}