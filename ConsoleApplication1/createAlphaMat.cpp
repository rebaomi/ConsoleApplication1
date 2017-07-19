#include "stdafx.h"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream>  
#include "windows.h"  

using namespace std;
using namespace cv;

void createAlphaMat(Mat &mat) 
{
	for (int i = 0;i < mat.rows;++i) 
	{
		for (int j = 0;j < mat.cols;++j)
		{
			Vec4b&rgba = mat.at<Vec4b>(i,j);
			rgba[0] = UCHAR_MAX;
			rgba[1] = saturate_cast<uchar>( (float(mat.cols - j)) / ((float)mat.cols ) * UCHAR_MAX);
			rgba[2] = saturate_cast<uchar>( (float(mat.rows - i)) / ((float)mat.rows ) * UCHAR_MAX);
			//rgba[3] = saturate_cast<uchar>(1);
			rgba[3] = saturate_cast<uchar>(1 * (rgba[1] + rgba[2]));

		}
	}
}

// 参考：http://blog.csdn.net/gdut2015go/article/details/49282031
int main_alpha() 
{
	Mat mat(480, 640, CV_8UC4);
	createAlphaMat(mat);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		imwrite("2.png", mat, compression_params);
		imshow("pngTu", mat);
		waitKey(0);
	}
	catch (runtime_error& ex) {
		return 1;
	}
	return 0;
	
}

// 参考：http://blog.csdn.net/wi162yyxq/article/details/53259404
int main2()
{
	// 这样我们读取的图片就有了四个通道，也就是BGRA四个通道，第四个就是Alpha通道，这个通道通过0-255的数值来代表该像素的透明度。
	// 我通过split函数分解矩阵的通道把png图片分解为B1矩阵，G1矩阵，R1矩阵，A1矩阵把背景图片分解为了B0矩阵，G0矩阵，R0矩阵。于是叠加的结果：令alpha = A1, 令beta = 255 - alpha。
	//Mat inimg = imread("dog.png", CV_LOAD_IMAGE_UNCHANGED);// 读取透明通道
	/*cout << (int)inimg.at<Vec4b>(0, 0)[0] << endl
		<< (int)inimg.at<Vec4b>(0, 0)[1] << endl
		<< (int)inimg.at<Vec4b>(0, 0)[2] << endl
		<< (int)inimg.at<Vec4b>(0, 0)[3] << endl;*/

	char str[16];
	Mat img1 = imread("dog.png"), src = imread("img.png", -1);
	Mat dst(img1, cvRect(0, 0, src.cols, src.rows));
	if (dst.channels() != 3 || src.channels() != 4)
	{
		return true;
	}


	vector<Mat> src_channels;
	vector<Mat> dst_channels;
	split(src, src_channels);
	split(dst, dst_channels);

	for (int i = 0;i < 3;i++)
	{
		dst_channels[i] = dst_channels[i].mul(255.0 / 1.0 - src_channels[3], 1.0 / 255.0);
		dst_channels[i] += src_channels[i].mul(src_channels[3], 1.0 / 255.0);
	}
	merge(dst_channels, dst);

	imshow("final", img1);
	waitKey(0);
	return 0;
}

int cvAdd4cMat_q(Mat &dst, Mat &src, double scale)
{
	if (dst.channels() != 3 || src.channels() != 4)
	{
		return true;
	}

	if (scale < 0.01)
	{
		return false;
	}

	vector<Mat> src_channels;
	vector<Mat> dst_channels;
	split(src, src_channels);
	split(dst, dst_channels);
	CV_Assert(src_channels.size() == 4 && dst_channels.size() == 3);

	if (scale < 1)
	{
		src_channels[3] *= scale;
		scale = 1;
	}
	for (int i = 0;i < 3;i++)
	{
		dst_channels[i] = dst_channels[i].mul(255.0 / scale - src_channels[3], scale / 255.0);
		dst_channels[i] += src_channels[i].mul(src_channels[3], scale / 255.0);
	}
	merge(dst_channels, dst);

	return true;
}

// 参考： http://blog.csdn.net/qq_28713863/article/details/72576725
