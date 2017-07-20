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
// 参考：http://blog.csdn.net/xiaowei_cqu/article/details/8987387
// 参考：http://blog.csdn.net/poem_qianmo/article/details/28261997

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

// 全局变量声明
Mat image0, image, gray, mask;// 定义原始图、目标图、灰度图、掩模图 
int ffillMode = 1;//漫水填充的模式
int loDiff = 20, upDiff = 20;//负差最大值、正差最大值
int connectivity = 4;//表示floodFill函数标识符低八位的连通值  
int isColor = true;//是否为彩色图的标识符布尔值  
bool useMask = false;//是否显示掩膜窗口的布尔值  
int newMaskVal = 255;//新的重新绘制的像素值  

//      描述：鼠标消息onMouse回调函数  
static void onMouse(int event, int x, int y, int, void*)
{
	// 若鼠标左键没有按下，便返回
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	//-------------------【<1>调用floodFill函数之前的参数准备部分】---------------  
	Point seed = Point(x, y);
	int lo = ffillMode == 0 ? 0 : loDiff;//空范围的漫水填充，此值设为0，否则设为全局的g_nLowDifference  
	int up = ffillMode == 0 ? 0 : upDiff;//空范围的漫水填充，此值设为0，否则设为全局的g_nUpDifference
	int flags = connectivity + (newMaskVal << 8) +
		//标识符的0~7位为g_nConnectivity，8~15位为g_nNewMaskVal左移8位的值，16~23位为CV_FLOODFILL_FIXED_RANGE或者0。
		(ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	//随机生成bgr值
	int b = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int g = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int r = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	
	Rect ccomp;//定义重绘区域的最小边界矩形区域  
    
    //在重绘区域像素的新值，若是彩色图模式，取Scalar(b, g, r)；若是灰度图模式，取Scalar(r*0.299 + g*0.587 + b*0.114)  
	Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	newVal = Scalar(255, 255, 255);
	Mat dst = isColor ? image : gray;//目标图的赋值 
	int area;

	//--------------------【<2>正式调用floodFill函数】----------------------------- 
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
	imwrite("ddd.png", dst);

	// Mat alphaDst;
	// cvtColor(dst, alphaDst, CV_RGB2RGBA);	
	// imwrite("alphaDst.png", alphaDst);
	cout << area << " pixels were repainted\n";
}

//-----------------------------------【main( )函数】--------------------------------------------    
//      描述：控制台应用程序的入口函数，我们的程序从这里开始    
//-----------------------------------------------------------------------------------------------    
int main_floodfill()
{
	//改变console字体颜色    
	//system("color 2F");

	//载入原图  
	char* filename = "10.jpg";
	image0 = imread(filename, 1);
	// image0 = imread(filename, -1);// imread后一个参数为-1表示读取带有alpha通道的图片

	if (image0.empty())
	{
		cout << "Image empty. Usage: ffilldemo <image_name>\n";
		return 0;
	}
	//显示帮助文字 
	floodfill_help();

	image0.copyTo(image);//拷贝源图到目标图
	cvtColor(image0, gray, CV_BGR2GRAY);//转换三通道的image0到灰度图  
	mask.create(image0.rows + 2, image0.cols + 2, CV_8UC1);//利用image0的尺寸来初始化掩膜mask  

	namedWindow("image", 0);// 效果图
	// namedWindow( "效果图",CV_WINDOW_AUTOSIZE );  

	//创建Trackbar  
	createTrackbar("lo_diff", "image", &loDiff, 255, 0);// 负差最大值
	createTrackbar("up_diff", "image", &upDiff, 255, 0);// 正差最大值

    //鼠标回调函数
	setMouseCallback("image", onMouse, 0);

	//循环轮询按键
	for (;;)
	{
		//先显示效果图 
		imshow("image", isColor ? image : gray);

		int c = waitKey(0);
		//判断ESC是否按下，若按下便退出
		if ((c & 255) == 27)
		{
			cout << "Exiting ...\n";
			break;
		}
		//根据按键的不同，进行各种操作 
		switch ((char)c)
		{
		case 'c':
			if (isColor)
			{
				cout << "Grayscale mode is set\n";// 切换彩色图/灰度图模式
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
			// 显示/隐藏掩膜窗口
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
			cout << "Original image is restored\n";// 恢复原始图像
			image0.copyTo(image);
			cvtColor(image, gray, CV_BGR2GRAY);
			mask = Scalar::all(0);
			break;
		case 's':
			cout << "Simple floodfill mode is set\n";// 使用空范围的漫水填充
			ffillMode = 0;
			break;
		case 'f':
			cout << "Fixed Range floodfill mode is set\n";// 使用渐变、固定范围的漫水填充
			ffillMode = 1;
			break;
		case 'g':
			cout << "Gradient (floating range) floodfill mode is set\n";//  使用渐变、浮动范围的漫水填充
			ffillMode = 2;
			break;
		case '4':
			cout << "4-connectivity mode is set\n";// 操作标识符的低八位使用4位的连接模式
			connectivity = 4;
			break;
		case '8':
			cout << "8-connectivity mode is set\n";// 操作标志符的低八位使用8位的连接模式
			connectivity = 8;
			break;
		}
	}

	return 0;
}

