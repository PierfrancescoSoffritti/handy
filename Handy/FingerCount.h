#pragma once

#include "opencv/cv.h"

using namespace cv;
using namespace std;

class FingerCount {
	public:
		FingerCount(void);
		Mat findFingersCount(Mat input_image, Mat frame);
	
	private:
		Scalar color_blue;
		Scalar color_green;
		Scalar color_red;
		Scalar color_black;
		Scalar color_white;
		Scalar color_yellow;
		Scalar color_purple;
		double findPointsDistance(Point a, Point b);
		vector<Point> compactOnNeighborhoodMedian(vector<Point> points, double max_neighbor_distance);
		double findAngle(Point a, Point b, Point c);
		bool isFinger(Point a, Point b, Point c, double limit_angle_inf, double limit_angle_sup, cv::Point palm_center, double distance_from_palm_tollerance);
		vector<Point> findClosestOnX(vector<Point> points, Point pivot);
		double findPointsDistanceOnX(Point a, Point b);
		void drawVectorPoints(Mat image, vector<Point> points, Scalar color, bool with_numbers);
};