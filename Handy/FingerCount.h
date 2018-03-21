#pragma once
#include "opencv/cv.h"

class FingerCount {
public:
	FingerCount(void);
	cv::Mat findFingersCount(cv::Mat input);
	double FingerCount::findPointsDistance(cv::Point a, cv::Point b);
	std::vector<cv::Point> FingerCount::compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance);
	double FingerCount::findAngle(cv::Point a, cv::Point b, cv::Point c);
	bool FingerCount::isFinger(cv::Point a, cv::Point b, cv::Point c, double limit_angle_inf, double limit_angle_sup, cv::Point palm_center, double distance_from_palm_tollerance);
};