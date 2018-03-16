#include "FingerCount.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DEBUG true
#define MAX_NEIGHBOR_DISTANCE 20

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
		input = imread("../res/hand.png", CV_LOAD_IMAGE_COLOR); 
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

	if (contours.size() <= 0)
		throw runtime_error("Contours size is negative?!\n");

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
		convexityDefects(contours[biggest_contour_index], hull_ints, defects);

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
	circle(contours_image, center_bounding_rect, 5, color_yellow, 2, 8);

	// we separate the defects
	vector<Point> start_points;
	vector<Point> far_points;
	vector<Point> end_points;
		
	for (int k = 0; k < defects.size(); k++) {
		
		// start points
		int start_idx = defects[k].val[0];
		start_points.push_back(contours[biggest_contour_index][start_idx]);
			
		// end points
		int end_idx = defects[k].val[1];
		end_points.push_back(contours[biggest_contour_index][end_idx]);
		
		// far points
		int far_idx = defects[k].val[2];
		far_points.push_back(contours[biggest_contour_index][far_idx]);
	}

	// we find and draw the median of those points
	vector<Point> start_points_medians = findNeighborhoodMedian(start_points);
	vector<Point> far_points_medians = findNeighborhoodMedian(far_points);
	vector<Point> end_points_medians = findNeighborhoodMedian(end_points);

	for(Point p : start_points_medians)
		circle(contours_image, p, 5, color_blue, 2, 8);
	for (Point p : far_points_medians)
		circle(contours_image, p, 5, color_red, 2, 8);
	for (Point p : end_points_medians)
		circle(contours_image, p, 5, color_white, 2, 8);

	return contours_image;
}

double FingerCount::findPointsDistance(cv::Point a, cv::Point b) {
	Point difference = a - b;
	return sqrt(difference.ddot(difference));
}

std::vector<cv::Point> FingerCount::findNeighborhoodMedian(std::vector<cv::Point> points) {
	vector<Point> median_points;

	if (points.size() == 0)
		throw runtime_error("You have passed an empty evector!\n");

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