#include "stdafx.h"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream>  
#include "windows.h"  

using namespace std;
using namespace cv;

void Sharpen(const Mat& myImage, Mat& Result)
{
	CV_Assert(myImage.depth() == CV_8U);  // 仅接受uchar图像

	Result.create(myImage.size(), myImage.type());
	const int nChannels = myImage.channels();

	for (int j = 1; j < myImage.rows - 1; ++j)//从第一行开始，而不是0;到rows-2行结束
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);//上一行
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);//下一行

		uchar* output = Result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * current[i]
				- current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
		}
	}

	//边缘处理。边缘没有卷积，所以边缘赋值为原图像对应像素的值。

	//先处理上和下
	uchar* output_top = Result.ptr<uchar>(0);
	const uchar* origin_top = myImage.ptr<uchar>(0);
	uchar* output_bottom = Result.ptr<uchar>(myImage.rows - 1);
	const uchar* origin_bottom = Result.ptr<uchar>(myImage.rows - 1);
	for (int i = 0; i < nChannels * myImage.cols - 1; ++i) {
		*output_top++ = *origin_top++;
		*output_bottom++ = *origin_bottom++;
	}

	//左和右 分别有nChannel
	for (int i = 0; i < myImage.rows; ++i) {
		const uchar* origin_left = myImage.ptr<uchar>(i);
		uchar* output_left = Result.ptr<uchar>(i);
		const uchar* origin_right = origin_left + nChannels*(myImage.cols - 1);
		uchar* output_right = output_left + nChannels*(myImage.cols - 1);
		for (int j = 0; j < nChannels; ++j) {
			*output_left++ = *origin_left++;
			*output_right++ = *origin_right++;
		}
	}
}

// 参考：http://www.cnblogs.com/korbin/p/5618559.html
int main_opencv(int argc, char* argv[]) {
	const char* path = "2.jpg";
	Mat img = imread(path);
	if (img.empty())
	{
		cout << "error";
		return -1;
	}
	imshow("原图像", img);

	Mat h_kern = (Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Mat h_mat;
	// 对图像进行卷积
	// 矩阵的掩码操作即对图像进行卷积。对图像卷积操作的意义为：邻近像素对（包括该像素自身）对新像素的影响；影响大小取决于卷积核对应位置值得大小。
	filter2D(img, h_mat, img.depth(), h_kern);
	imshow("水平方向边缘提取", h_mat);

	Mat v_kern = (Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
	Mat v_mat;
	filter2D(img, v_mat, img.depth(), v_kern);

	imshow("线性非均值滤波2", v_mat);

	Mat l_kern = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	Mat l_mat;

	filter2D(img, l_mat, img.depth(), l_kern);
	imshow("lapulasi", l_mat);

	/****************************Roberts**************************/
	Mat Roberts_kern_x = (Mat_<float>(2, 2) << -1, 0,
		0, 1);

	Mat Roberts_kern_y = (Mat_<float>(2, 2) << 0, 1,
		-1, 0);

	Mat Roberts_Mat_x, Roberts_Mat_y, Roberts_Mat;

	filter2D(img, Roberts_Mat_x, img.depth(), Roberts_kern_x);
	filter2D(img, Roberts_Mat_y, img.depth(), Roberts_kern_y);
	Mat Roberts_abs_x, Roberts_abs_y;
	convertScaleAbs(Roberts_Mat_x, Roberts_abs_x);
	convertScaleAbs(Roberts_Mat_y, Roberts_abs_y);
	addWeighted(Roberts_abs_x, 0.5, Roberts_abs_y, 0.5, 0, Roberts_Mat);
	imshow("Roberts", Roberts_Mat);
	/****************************Roberts**************************/

	/****************************Sobel**************************/

	Mat Sobel_Mat_x, Sobel_Mat_y, Sobel_Mat;
	Sobel(img, Sobel_Mat_x, img.depth(), 1, 0);
	Sobel(img, Sobel_Mat_y, img.depth(), 0, 1);
	convertScaleAbs(Sobel_Mat_x, Sobel_Mat_x);
	convertScaleAbs(Sobel_Mat_y, Sobel_Mat_y);
	addWeighted(Sobel_Mat_x, 0.5, Sobel_Mat_y, 0.5, 0, Sobel_Mat);
	imshow("Sobel", Sobel_Mat);

	/****************************Sobel**************************/

	/****************************Priwitt**************************/
	Mat Priwitt_kern_x = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	Mat Priwitt_kern_y = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);

	Mat Priwitt_Mat_x, Priwitt_Mat_y, Priwitt_Mat;
	filter2D(img, Priwitt_Mat_x, img.depth(), Priwitt_kern_x);
	filter2D(img, Priwitt_Mat_y, img.depth(), Priwitt_kern_y);
	convertScaleAbs(Priwitt_Mat_x, Priwitt_Mat_x);
	convertScaleAbs(Priwitt_Mat_y, Priwitt_Mat_y);
	addWeighted(Priwitt_Mat_x, 0.5, Priwitt_Mat_y, 0.5, 0, Priwitt_Mat);
	imshow("Peiwitt", Priwitt_Mat);

	// 计算直方图
	//把BGR图像分割为单通道图像
	vector<Mat> bgr_planes;

	split(img, bgr_planes);
	cout << bgr_planes.size() << endl;
	//计算直方图
	vector<Mat> hist_image;
	hist_image.resize(3);

	//直方图统计像素类别数
	const int histSize[] = { 255 };
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	const int channels[] = { 0 };

	calcHist(&bgr_planes[0], 1, channels, Mat(), hist_image[0], 1, histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, channels, Mat(), hist_image[1], 1, histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, channels, Mat(), hist_image[2], 1, histSize, &histRange, uniform, accumulate);
	/// 将直方图高度归一化到范围 [ 0, histImage.rows ]
	normalize(hist_image[0], hist_image[0], 0, hist_image[0].rows, NORM_MINMAX, -1, Mat());
	normalize(hist_image[1], hist_image[1], 0, hist_image[1].rows, NORM_MINMAX, -1, Mat());
	normalize(hist_image[2], hist_image[2], 0, hist_image[2].rows, NORM_MINMAX, -1, Mat());

	// 创建直方图画布  
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize[0]);//每个像素宽度

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// 在直方图画布上画出直方图。Mat坐标系，原点在左上角，习惯用的坐标系原点在左下角，因此高度要调整。即画布height - y
	for (int i = 1; i < histSize[0]; i++)
	{
		//R
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist_image[2].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist_image[2].at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		//G
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist_image[1].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist_image[1].at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		//B
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist_image[0].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist_image[0].at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// 显示直方图
	imshow("Hist", histImage);

	// 直方图均衡化是用来调整图像对比度的方法。它的思想是把某些像素值集中的区间分配到其他像素值上，在直方图上的表现就是拉伸了直方图，但是直方图的面积（图像总亮度）未改变，只是重新分配了而已。
	// 经过直方图均衡化后，图像的像素间差异减少，一些图像细节可能消失或减弱。
	// 三个通道直方图均衡化  
	vector<Mat> rgb_planes;
	split(img, rgb_planes);
	equalizeHist(rgb_planes[0], rgb_planes[0]);
	equalizeHist(rgb_planes[1], rgb_planes[1]);
	equalizeHist(rgb_planes[2], rgb_planes[2]);

	Mat new_img;
	merge(rgb_planes, new_img);
	imshow("直方图均衡化", new_img);

	// 蜕化
	Mat result;
	Sharpen(img, result);
	imshow("蜕化", result);

	//均值滤波
	Mat blur_mat;
	blur(img, blur_mat, Size(3, 3));
	imshow("均值滤波", blur_mat);

	//线性非均值滤波
	Mat kern = (Mat_<float>(3, 3) << 1, 2, 1,
		2, 4, 2,
		1, 2, 1) / 16;
	Mat blur2_mat;
	filter2D(img, blur2_mat, img.depth(), kern);
	imshow("线性非均值滤波", blur2_mat);

	//中值滤波
	Mat memedian_mat;
	medianBlur(img, memedian_mat, 3);
	imshow("中值滤波", memedian_mat);

	//高斯滤波
	Mat Gaussian_mat;
	GaussianBlur(img, Gaussian_mat, Size(3, 3), 0, 0);
	imshow("高斯滤波", Gaussian_mat);

	//双边滤波

	Mat bilateral_mat;
	bilateralFilter(img, bilateral_mat, 25, 25 * 2, 25 / 2);
	imshow("双边滤波", bilateral_mat);


	waitKey();
	return 0;
}

// 魔棒工具 floodfill

static void floodfill_help() 
{
	cout << "\nThis program demonstrated the floodFill() function\n"
		"Call:\n"
		"./ffilldemo [image_name -- Default: fruits.jpg]\n" << endl;

	cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tc - switch color/grayscale mode\n"
		"\tm - switch mask mode\n"
		"\tr - restore the original image\n"
		"\ts - use null-range floodfill\n"
		"\tf - use gradient floodfill with fixed(absolute) range\n"
		"\tg - use gradient floodfill with floating(relative) range\n"
		"\t4 - use 4-connectivity mode\n"
		"\t8 - use 8-connectivity mode\n" << endl;
}

Mat image0, image, gray, mask;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

static void onMouse(int event, int x, int y, int, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	Point seed = Point(x, y);
	int lo = ffillMode == 0 ? 0 : loDiff;
	int up = ffillMode == 0 ? 0 : upDiff;
	int flags = connectivity + (newMaskVal << 8) +
		(ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	int b = (unsigned)theRNG() & 255;
	int g = (unsigned)theRNG() & 255;
	int r = (unsigned)theRNG() & 255;
	Rect ccomp;

	Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	Mat dst = isColor ? image : gray;
	int area;

	if (useMask)
	{
		threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
			Scalar(up, up, up), flags);
		imshow("mask", mask);
	}
	else
	{
		area = floodFill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
			Scalar(up, up, up), flags);
	}

	imshow("image", dst);
	cout << area << " pixels were repainted\n";
}


int main()
{
	char* filename = "2.jpg";
	image0 = imread(filename, 1);

	if (image0.empty())
	{
		cout << "Image empty. Usage: ffilldemo <image_name>\n";
		return 0;
	}
	floodfill_help();
	image0.copyTo(image);
	cvtColor(image0, gray, CV_BGR2GRAY);
	mask.create(image0.rows + 2, image0.cols + 2, CV_8UC1);

	namedWindow("image", 0);
	createTrackbar("lo_diff", "image", &loDiff, 255, 0);
	createTrackbar("up_diff", "image", &upDiff, 255, 0);

	setMouseCallback("image", onMouse, 0);

	for (;;)
	{
		imshow("image", isColor ? image : gray);

		int c = waitKey(0);
		if ((c & 255) == 27)
		{
			cout << "Exiting ...\n";
			break;
		}
		switch ((char)c)
		{
		case 'c':
			if (isColor)
			{
				cout << "Grayscale mode is set\n";
				cvtColor(image0, gray, CV_BGR2GRAY);
				mask = Scalar::all(0);
				isColor = false;
			}
			else
			{
				cout << "Color mode is set\n";
				image0.copyTo(image);
				mask = Scalar::all(0);
				isColor = true;
			}
			break;
		case 'm':
			if (useMask)
			{
				destroyWindow("mask");
				useMask = false;
			}
			else
			{
				namedWindow("mask", 0);
				mask = Scalar::all(0);
				imshow("mask", mask);
				useMask = true;
			}
			break;
		case 'r':
			cout << "Original image is restored\n";
			image0.copyTo(image);
			cvtColor(image, gray, CV_BGR2GRAY);
			mask = Scalar::all(0);
			break;
		case 's':
			cout << "Simple floodfill mode is set\n";
			ffillMode = 0;
			break;
		case 'f':
			cout << "Fixed Range floodfill mode is set\n";
			ffillMode = 1;
			break;
		case 'g':
			cout << "Gradient (floating range) floodfill mode is set\n";
			ffillMode = 2;
			break;
		case '4':
			cout << "4-connectivity mode is set\n";
			connectivity = 4;
			break;
		case '8':
			cout << "8-connectivity mode is set\n";
			connectivity = 8;
			break;
		}
	}

	return 0;
}