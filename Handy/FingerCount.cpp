#include "FingerCount.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DEBUG false
#define LIMIT_ANGLE_SUP 60
#define LIMIT_ANGLE_INF 5
#define BOUNDING_RECT_FINGER_SIZE_SCALING 0.35
#define BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING 0.05
#define LIMIT_MIN_DISTANCE_FROM_PALM 120

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
		printf("Error findHandContours, could not open the input image!\n");

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
		printf("Error findHandContours, could not find the biggest contour!\n");
	
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
		defect_points.push_back(contours[biggest_contour_index][defects[i].val[0]]); // start points
		defect_points.push_back(contours[biggest_contour_index][defects[i].val[2]]); // far points
	}

	vector<Point> filtered_defects_points = compactOnNeighborhoodMedian(defect_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);
	
	vector<Point> extr_points;
	vector<Point> depr_points;

	for (Point p : filtered_defects_points) {
		if (findPointsDistance(p, center_bounding_rect) > bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING)
			extr_points.push_back(p);
		else
			depr_points.push_back(p);
	}

	vector<Point> filtered_extr_points = compactOnNeighborhoodMedian(extr_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 2); // we apply a stronger filter
	vector<Point> filtered_depr_points = compactOnNeighborhoodMedian(depr_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);

	vector<Point> all;
	for (int i = 0; i < filtered_extr_points.size() && i < filtered_depr_points.size(); i++) {
		all.push_back(filtered_extr_points[i]);
		all.push_back(filtered_depr_points[i]);
	}

	// we draw all the points, now supposedly filtered and ordered
	for (int i = 0; i < all.size(); i++) {
		circle(contours_image, all[i], 5, color_red, 2, 8);
		putText(contours_image, to_string(i), all[i], FONT_HERSHEY_PLAIN, 3, color_red);
	}

	if (all.size() > 3) {
		vector<int> fingertip_point_index;

		// first point
		if (isFinger(all[all.size() - 1], all[0], all[1], LIMIT_ANGLE_INF, LIMIT_ANGLE_SUP, center_bounding_rect, bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING))
			fingertip_point_index.push_back(0);

		// every other point
		for (int i = 1; i < all.size() - 1 && fingertip_point_index.size() < 5; i++) {
			if ((i % 2) == 0) {
				if (isFinger(all[i - 1], all[i], all[i + 1], LIMIT_ANGLE_INF, LIMIT_ANGLE_SUP, center_bounding_rect, bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING))
					fingertip_point_index.push_back(i);
			}
		}

		// we draw the detected points
		vector<Point> fingers;
		for (int i : fingertip_point_index) {
			fingers.push_back(all[i]);
		}
		
		vector<Point> filtered_fingers = compactOnNeighborhoodMedian(fingers, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 3);
		for (Point p : filtered_fingers) {
			circle(contours_image, p, 5, color_yellow, 2, 8);
		}

		// we draw the number of finers found
		putText(contours_image, to_string(fingers.size()), center_bounding_rect, FONT_HERSHEY_PLAIN, 3, color_white);
	}

	return contours_image;
}

double FingerCount::findPointsDistance(cv::Point a, cv::Point b) {
	Point difference = a - b;
	return sqrt(difference.ddot(difference));
}

std::vector<cv::Point> FingerCount::compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance) {
	vector<Point> median_points;
	
	if (points.size() == 0) {
		printf("Error compactOnNeighborhoodMedian, points.size() zero.\n");
		return median_points;
	}

	if (max_neighbor_distance <= 0) {
		printf("Error compactOnNeighborhoodMedian, max_neighbor_distance less or equal zero.\n");
		return median_points;
	}

	// we start with the first point
	Point reference = points[0];
	Point median = points[0];

	for (int i = 1; i < points.size(); i++) {
		if (findPointsDistance(reference, points[i]) > max_neighbor_distance) {
			
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

bool FingerCount::isFinger(cv::Point a, cv::Point b, cv::Point c, double limit_angle_inf, double limit_angle_sup, Point palm_center, double min_distance_from_palm) {
	if (min_distance_from_palm < LIMIT_MIN_DISTANCE_FROM_PALM)
		return false;
	
	double angle = findAngle(a, b, c);
	if (angle > limit_angle_sup || angle < limit_angle_inf)
		return false;

	int delta_y_1 = b.y - a.y;
	int delta_y_2 = b.y - c.y;
	if (delta_y_1 > 0 && delta_y_2 > 0)
		return false;

	double distance_from_palm = findPointsDistance(b, palm_center);
	if (distance_from_palm < min_distance_from_palm)
		return false;

	return true;
}