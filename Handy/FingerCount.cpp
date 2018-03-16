#include "FingerCount.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DEBUG false
#define MAX_NEIGHBOR_DISTANCE 20
#define LIMIT_ANGLE 40

using namespace cv;
using namespace std;

FingerCount::FingerCount(void) {}

cv::Mat FingerCount::findHandContours(cv::Mat input) {
	
	// colors
	Scalar color_blue(255, 0, 0);
	Scalar color_green(0, 255, 0);
	Scalar color_red(0, 0, 255);
	Scalar color_white(255, 255, 255);
	Scalar color_yellow(0, 255, 255);

	if (DEBUG) {
		input = imread("../res/handy.png", CV_LOAD_IMAGE_COLOR); 
		cvtColor(input, input, CV_BGR2GRAY);
	}

	// check if the source image is good
	if (input.empty())
		throw runtime_error("Could not open the input image!\n");

	// image returned by this function
	Mat contours_image = Mat::zeros(input.size(), CV_8UC3);

	// we work only on the 1 channel result, since this function is called inside a loop we are not sure that this is always the case
	if (input.channels() != 1)
		return contours_image;

	// find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// we need at least one contour to work
	if (contours.size() <= 0)
		return contours_image;

	// find the biggest contour (let's suppose it's our hand)
	int biggest_contour_index = -1;
	double biggest_area = 0;

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i], false);
		if (area > biggest_area) {
			biggest_area = area;
			biggest_contour_index = i;
		}
	}

	if (biggest_contour_index < 0)
		throw runtime_error("Could not find the biggest contour!\n");

	// find the convex hull object for each contour and the defects, two different data structure are needed by the OpenCV api
	vector<Point> hull_points; // for drawing the convex hull
	vector<int> hull_ints; // for finding the defects
	vector<Vec4i> defects;
	Rect bounding_rectangle; // needed for filtering defects later on

	convexHull(Mat(contours[biggest_contour_index]), hull_points, true);
	convexHull(Mat(contours[biggest_contour_index]), hull_ints, false);

	// we need at least 3 points to find the defects
	if (hull_ints.size() > 3)
		convexityDefects(Mat(contours[biggest_contour_index]), hull_ints, defects);

	// we bound the convex hull
	bounding_rectangle = boundingRect(Mat(hull_points));
	
	// we draw the contour, the convex hull and the bounding rectangle
	drawContours(contours_image, contours, biggest_contour_index, color_green, 2, 8, hierarchy);
	polylines(contours_image, hull_points, true, color_blue);
	rectangle(contours_image, bounding_rectangle.tl(), bounding_rectangle.br(), color_red, 2, 8, 0);

	// we find the center of the bounding rectangle, this should approximately also be the center of the hand
	Point center_bounding_rect(
		(bounding_rectangle.tl().x + bounding_rectangle.br().x)/2, 
		(bounding_rectangle.tl().y + bounding_rectangle.br().y)/2
	);
	circle(contours_image, center_bounding_rect, 5, color_blue, 2, 8);

	// we separate the defects keeping only the ones of intrest
	vector<Point> defect_points;
		
	for (int i = 0; i < defects.size(); i++) {
		
		// start points
		defect_points.push_back(contours[biggest_contour_index][defects[i].val[0]]);
			
		// far points
		defect_points.push_back(contours[biggest_contour_index][defects[i].val[2]]);
	}

	vector<Point> filtered_points = findNeighborhoodMedian(defect_points);
		
	//for (int i = 0; i < filtered_points.size(); i++) {
	//	char str[200];
	//	sprintf_s(str, "%d", i);
	//	if ( i % 2 == 0)
	//		circle(contours_image, filtered_points[i], 5, color_white, 2, 8);
	//	else
	//		circle(contours_image, filtered_points[i], 5, color_red, 2, 8);
	//	putText(contours_image, str, filtered_points[i], FONT_HERSHEY_PLAIN, 2, color_white);
	//}
	
	vector<int> fingertip_point_index;
	
	// first point
	if (findAngle(filtered_points[filtered_points.size() - 1], filtered_points[0], filtered_points[1]) < LIMIT_ANGLE)
		fingertip_point_index.push_back(0);

	// every other point
	for (int i = 1; i < filtered_points.size() - 1; i++) {
		if ((i % 2) == 0) {
			if (findAngle(filtered_points[i - 1], filtered_points[i], filtered_points[i + 1]) < LIMIT_ANGLE)
				fingertip_point_index.push_back(i);
		}
	}
	
	for (int i : fingertip_point_index) {
		circle(contours_image, filtered_points[i], 5, color_yellow, 2, 8);
	}

	return contours_image;
}

double FingerCount::findPointsDistance(cv::Point a, cv::Point b) {
	Point difference = a - b;
	return sqrt(difference.ddot(difference));
}

std::vector<cv::Point> FingerCount::findNeighborhoodMedian(std::vector<cv::Point> points) {
	if (points.size() == 0)
		throw runtime_error("You have passed an empty evector!\n");

	vector<Point> median_points;

	// we start with the first point
	Point reference = points[0];
	Point median = points[0];

	for (int i = 1; i < points.size(); i++) {
		if (findPointsDistance(reference, points[i]) > MAX_NEIGHBOR_DISTANCE) {
			
			// the point is not in range, we save the median
			median_points.push_back(median);

			// we swap the reference
			reference = points[i];
			median = points[i];
		}
		else
			median = (points[i] + median) * 0.5;
	}

	// last median
	median_points.push_back(median);

	return median_points;
}

double FingerCount::findAngle(cv::Point a, cv::Point b, cv::Point c) {
	double ab = findPointsDistance(a, b);
	double bc = findPointsDistance(b, c);
	double ac = findPointsDistance(a, c);
	return acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * 180 / CV_PI;
}