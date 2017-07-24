#pragma once

#ifndef SHAREDMATTING_H
#define SHAREDMATTING_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

struct labelPoint
{
	int x;
	int y;
	int label;
};

struct Tuple
{
	Scalar f;
    Scalar b;
	double sigmaf; // sigma ±ê×¼²î
	double sigmab;

	int flag;
};

struct Ftuple
{
	Scalar f;
	Scalar b;
	double alphar;
	double confidence;
};

class SharedMatting
{
public:
	SharedMatting();
	~SharedMatting();

	void loadImage(char * filename);
	void loadTrimap(char * filename);
	void expandKnown();
	void sample(Point p, vector<Point>& f, vector<Point>& b);
	void gathering();
	void refineSample();
	void localSmooth();
	void solveAlpha();
	void save(char * filename);
	void Sample(vector<vector<Point>> &F, vector<vector<Point>> &B);
	void getMatte();
	void release();

	double mP(int i, int j, Scalar f, Scalar b);
	double nP(int i, int j, Scalar f, Scalar b);
	double eP(int i1, int j1, int i2, int j2);
	double pfP(Point p, vector<Point>& f, vector<Point>& b);
	double aP(int i, int j, double pf, Scalar f, Scalar b);
	double gP(Point p, Point fp, Point bp, double pf);
	double gP(Point p, Point fp, Point bp, double dpf, double pf);
	double dP(Point s, Point d);
	double sigma2(Point p);
	double distanceColor2(Scalar cs1, Scalar cs2);
	double comalpha(Scalar c, Scalar f, Scalar b);

private:
	Mat pImg;
	Mat trimap;
	Mat matte;

	vector<Point> uT;
	vector<struct Tuple> tuples;
	vector<struct Ftuple> ftuples;

	int height;
	int width;
	int kI;
	int kG;
	int ** unknownIndex;
	int ** tri;
	int ** alpha;
	double kC;

	int step;
	int channels;
	uchar* data;
};




#endif

