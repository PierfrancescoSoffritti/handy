#pragma once
#include "opencv/cv.h"

class FingerCount {
public:
	FingerCount(void);
	cv::Mat findHandContours(cv::Mat input);
	double FingerCount::findPointsDistance(cv::Point a, cv::Point b);
	std::vector<cv::Point> FingerCount::findNeighborhoodMedian(std::vector<cv::Point> points);
	double FingerCount::findAngle(cv::Point a, cv::Point b, cv::Point c);
};