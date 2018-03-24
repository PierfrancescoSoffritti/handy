#include "FingerCount.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define LIMIT_ANGLE_SUP 50
#define LIMIT_ANGLE_INF 10
#define BOUNDING_RECT_FINGER_SIZE_SCALING 0.3
#define BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING 0.1

using namespace cv;
using namespace std;

FingerCount::FingerCount(void) {}

cv::Mat FingerCount::findFingersCount(cv::Mat input) {

	// colors
	Scalar color_blue(255, 0, 0);
	Scalar color_green(0, 255, 0);
	Scalar color_red(0, 0, 255);
	Scalar color_white(255, 255, 255);
	Scalar color_yellow(0, 255, 255);

	// check if the source image is good
	if (input.empty())
		printf("Error findFingersCount: Could not open the input image!\n");

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
		printf("Error findFingersCount: Could not find biggest contour!\n");

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
		(bounding_rectangle.tl().x + bounding_rectangle.br().x) / 2,
		(bounding_rectangle.tl().y + bounding_rectangle.br().y) / 2
	);
	circle(contours_image, center_bounding_rect, 5, color_white, 2, 8);

	// we separate the defects keeping only the ones of intrest
	vector<Point> start_points;
	vector<Point> far_points;

	for (int i = 0; i < defects.size(); i++) {
		start_points.push_back(contours[biggest_contour_index][defects[i].val[0]]); // start points
		if (findPointsDistance(contours[biggest_contour_index][defects[i].val[2]], center_bounding_rect) < bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING)
			far_points.push_back(contours[biggest_contour_index][defects[i].val[2]]); // far points
	}

	// we compact them on their medians
	vector<Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);
	vector<Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);

	// we draw what found
	for (int i = 0; i < filtered_start_points.size(); i++) {
		circle(contours_image, filtered_start_points[i], 5, color_blue, 2, 8);
		putText(contours_image, to_string(i), filtered_start_points[i], FONT_HERSHEY_PLAIN, 3, color_blue);
	}
	for (int i = 0; i < filtered_far_points.size(); i++) {
		circle(contours_image, filtered_far_points[i], 5, color_red, 2, 8);
		putText(contours_image, to_string(i), filtered_far_points[i], FONT_HERSHEY_PLAIN, 3, color_red);
	}

	// now we try to find the fingers
	if (filtered_far_points.size() > 2) {
		vector<int> fingertip_index;

		for (int i = 0; i < filtered_start_points.size(); i++) {
			vector<Point> close = findClosestOnX(filtered_far_points, filtered_start_points[i]);

			if (close.size() == 2) {
				if (isFinger(close[0], filtered_start_points[i], close[1], LIMIT_ANGLE_INF, LIMIT_ANGLE_SUP, center_bounding_rect, bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING))
					fingertip_index.push_back(i);
			}
		}

		// we have at most five fingers usually :)
		while (fingertip_index.size() > 5)
			fingertip_index.pop_back();

		//if (fingertip_index.size() > 1) {
		//	if (findPointsDistanceOnX(filtered_start_points[fingertip_index[0]], filtered_start_points[fingertip_index[fingertip_index.size() - 1]]) < bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 2)		
		//		fingertip_index.pop_back();
		//	
		//	for (int i = 1; i < fingertip_index.size() - 1; i++) {
		//		if (findPointsDistanceOnX(filtered_start_points[fingertip_index[i]], filtered_start_points[fingertip_index[i + 1]]) < bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 10)	
		//			fingertip_index.erase(fingertip_index.begin() + i);
		//	}
		//}

		// we draw the fingers found
		for (int i : fingertip_index)
			circle(contours_image, filtered_start_points[i], 5, color_yellow, 2, 8);

		putText(contours_image, to_string(fingertip_index.size()), center_bounding_rect, FONT_HERSHEY_PLAIN, 3, color_yellow);
	}

	return contours_image;
}

double FingerCount::findPointsDistance(cv::Point a, cv::Point b) {
	Point difference = a - b;
	return sqrt(difference.ddot(difference));
}

std::vector<cv::Point> FingerCount::compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance) {
	vector<Point> median_points;
	
	if (points.size() == 0)		
		return median_points;

	if (max_neighbor_distance <= 0)
		return median_points;

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
			median = (points[i] + median) / 2;
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

std::vector<cv::Point> FingerCount::findClosestOnX(std::vector<cv::Point> points, cv::Point pivot) {
	vector<Point> to_return;

	if (points.size() == 0)
		return to_return;

	double distance_x_1 = DBL_MAX;
	double distance_1 = DBL_MAX;
	double distance_x_2 = DBL_MAX;
	double distance_2 = DBL_MAX;
	int index_found = 0;

	for (int i = 0; i < points.size(); i++) {
		double distance_x = findPointsDistanceOnX(pivot, points[i]);
		double distance = findPointsDistance(pivot, points[i]);

		if (distance_x < distance_x_1 && distance_x != 0 && distance <= distance_1) {
			distance_x_1 = distance_x;
			distance_1 = distance;
			index_found = i;
		}
	}

	to_return.push_back(points[index_found]);

	for (int i = 0; i < points.size(); i++) {
		double distance_x = findPointsDistanceOnX(pivot, points[i]);
		double distance = findPointsDistance(pivot, points[i]);

		if (distance_x < distance_x_2 && distance_x != 0 && distance <= distance_2 && distance_x != distance_x_1) {
			distance_x_2 = distance_x;
			distance_2 = distance;
			index_found = i;
		}
	}

	to_return.push_back(points[index_found]);

	return to_return;
}

double FingerCount::findPointsDistanceOnX(cv::Point a, cv::Point b) {
	double to_return = 0.0;

	if (a.x > b.x)
		to_return = a.x - b.x;
	else
		to_return = b.x - a.x;

	return to_return;
}
