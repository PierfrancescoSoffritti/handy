#include "FingerCount.h"

#include "opencv2/imgproc.hpp"

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

	// find the convex hull object for each contour and the defects, two different data structure are needed by the OpenCV api
	vector<vector<Point>> hulls_point(contours.size()); // for drawing the hulls
	vector<vector<int>> hulls_int(contours.size()); // for finding the defects
	vector<vector<Vec4i>> defects(contours.size());
	vector<Rect> bounding_rectangle(contours.size()); // for filtering defects

	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hulls_point[i], false);
		convexHull(Mat(contours[i]), hulls_int[i], false);
		// is this if really needed? check
		if (hulls_int[i].size() > 3)
			convexityDefects(contours[i], hulls_int[i], defects[i]);

		// we bound the convex hull
		bounding_rectangle[i] = boundingRect(Mat(hulls_point[i]));
	}

	// find the biggest contour (let's suppose it's our hand)
	double biggest_area = 0;
	int biggest_contour_index = -1;

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i], false);
		if (area > biggest_area) {
			biggest_area = area;
			biggest_contour_index = i;
		}
	}

	// show all contours
	//for (int i = 0; i< contours.size(); i++) {
	//	drawContours(contours_image, contours, i, color_green, 2, 8, hierarchy);
	//	drawContours(contours_image, hulls_point, i, color_blue, 2, 8, hierarchy);
	//}

	// show only the biggest contour
	if (biggest_contour_index >= 0) {
		// we draw the contour
		drawContours(contours_image, contours, biggest_contour_index, color_green, 2, 8, hierarchy);

		// we draw the convex hull
		drawContours(contours_image, hulls_point, biggest_contour_index, color_blue, 2, 8, hierarchy);

		// we draw the defects
		for (int k = 0; k < defects[biggest_contour_index].size(); k++) {
			// start points
			int start_idx = defects[biggest_contour_index][k].val[0];
			Point start_points(contours[biggest_contour_index][start_idx]);
			circle(contours_image, start_points, 5, color_blue, 2, 8);

			// end points
			int end_idx = defects[biggest_contour_index][k].val[1];
			Point end_points(contours[biggest_contour_index][end_idx]);
			circle(contours_image, end_points, 5, color_white, 2, 8);

			// far points
			int far_idx = defects[biggest_contour_index][k].val[2];
			Point far_points(contours[biggest_contour_index][far_idx]);
			circle(contours_image, far_points, 5, color_red, 2, 8);
		}

		// we draw the bounding rectangle
		rectangle(contours_image, bounding_rectangle[biggest_contour_index].tl(), bounding_rectangle[biggest_contour_index].br(), color_red, 2, 8, 0);

		/* TODO: now we have all the info needed to filter out the defects, we need to find the angle (or the distance)
		between the neighbouring vertices of the hull and the defects and then deciding if they are usefull or not.. */

		// we find the center of the bounding rectangle, this should approximately also be the center of the hand
		Point center_bounding_rect(
			(bounding_rectangle[biggest_contour_index].tl().x + bounding_rectangle[biggest_contour_index].br().x)/2, 
			(bounding_rectangle[biggest_contour_index].tl().y + bounding_rectangle[biggest_contour_index].br().y)/2
		);
		circle(contours_image, center_bounding_rect, 5, color_yellow, 2, 8);

	}	
	else {
		throw runtime_error("Could not find the biggest contour!\n");
	}

	return contours_image;
}