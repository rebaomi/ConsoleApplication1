#include "stdafx.h"  
#include "sharedmatting.h"
#include "guidedfilter.h"
#include "globalmatting.h"
#include <string>

using namespace std;

int main_alphamatting()
{
	char fileAddr[64] = { 0 };

	for (int n = 1; n < 28; ++n) {
		SharedMatting sm;

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/input.png", n / 10, n % 10);
		sm.loadImage("C:/Users/Chuangkit_Developer7/Desktop/1-input.png");

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/trimap.png", n / 10, n % 10);
		sm.loadTrimap("C:/Users/Chuangkit_Developer7/Desktop/1-trimap.png");

		sm.solveAlpha();

		//sprintf(fileAddr, "C:/Users/Chuangkit_Developer7/Desktop/result.png", n / 10, n % 10);
		sm.save("C:/Users/Chuangkit_Developer7/Desktop/1-result.png");
	}

	return 0;
}

// not success
int main_globalmatting()
{
	Mat image = imread("C:/Users/Chuangkit_Developer7/Desktop/input.png", CV_LOAD_IMAGE_COLOR);
	Mat trimap = imread("C:/Users/Chuangkit_Developer7/Desktop/trimap.png", CV_LOAD_IMAGE_GRAYSCALE);

	// (optional) exploit the affinity of neighboring pixels to reduce the 
	// size of the unknown region. please refer to the paper
	// 'Shared Sampling for Real-Time Alpha Matting'.
	expansionOfKnownRegions(image, trimap, 9);

	Mat foreground, alpha;
	globalMatting(image, trimap, foreground, alpha);

	// filter the result with fast guided filter
	alpha = guidedFilter(image, alpha, 10, 1e-5);
	for (int x = 0; x < trimap.cols; ++x)
		for (int y = 0; y < trimap.rows; ++y)
		{
			if (trimap.at<uchar>(y, x) == 0)
				alpha.at<uchar>(y, x) = 0;
			else if (trimap.at<uchar>(y, x) == 255)
				alpha.at<uchar>(y, x) = 255;
		}

	imwrite("GT04-alpha.png", alpha);

	return 0;
}
// not success
int main_guidedfilter() {
	//Mat cat = imread("C:/Users/Chuangkit_Developer7/Desktop/1-input.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat	 p = cat;
	//int r = 4;
	//double eps = 0.2 * 0.2;
	//eps *= 255 * 255;
	//Mat  q = guidedFilter(cat, p, r, eps);
	Mat I = imread("C:/Users/Chuangkit_Developer7/Desktop/toy-mask.png", CV_LOAD_IMAGE_COLOR);
	Mat p = imread("C:/Users/Chuangkit_Developer7/Desktop/toy.png", CV_LOAD_IMAGE_GRAYSCALE);

	int r = 60;
	double eps = 1e-6;

	eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]

	Mat q = guidedFilter(I, p, r, eps);
	
	imshow("result", q);
	waitKey(0);
	return 0;
}