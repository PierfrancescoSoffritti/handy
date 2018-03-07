#include "FaceDetector.h"
#include"opencv2\opencv.hpp"

using namespace cv;
using namespace std;

Rect getFaceRect(Mat input);
Mat getSkinInstogramFromFace(Mat input, int bins);
void drawHistogram(Mat input, int bins);


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

Mat FaceDetector::getSkinHistogram(Mat input) {
	Rect faceRect = getFaceRect(input);

	Mat face = Mat(input, faceRect);
	//imshow("face", face);

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

Mat getSkinInstogramFromFace(Mat input, int bins) {
	/*
	List of the dims channels used to compute the histogram.
	The first array channels are numerated from 0 to images[0].channels()-1,
	the second array channels are counted from images[0].channels() to images[0].channels() + images[1].channels()-1,
	and so on.
	*/
	int channels[] = { 0, 1 };

	Mat histogram;
	float hue_range[] = { 120, 200 };
	const float* ranges[] = { hue_range };

	// Get the Histogram and normalize it
	calcHist(&input, 1, channels, Mat(), histogram, 1, &bins, ranges, true, false);
	normalize(histogram, histogram, 0, 255, NORM_MINMAX, -1, Mat());

	drawHistogram(histogram, bins);

	return histogram;
}

void drawHistogram(Mat histogram, int bins) {
	int width = 400, height = 400;
	int bin_width = cvRound( width / bins);
	Mat histogramImage = Mat::zeros(width, height, CV_8UC3);

	for (int i = 0; i < bins; i++)
		rectangle(
			histogramImage,
			Point(i*bin_width, height),
			Point( (i + 1) * bin_width, height - cvRound( histogram.at<float>(i)*height / 255.0 ) ),
			Scalar(0, 0, 255),
			-1
		);

	imshow("faceHistogramImage", histogramImage);
}