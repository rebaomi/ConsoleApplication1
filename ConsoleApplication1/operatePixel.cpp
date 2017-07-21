#include "stdafx.h"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream>  
#include "windows.h"  

using namespace std;
using namespace cv;

/*
 【OpenCV】访问Mat中每个像素的值（新）
  参考：http://blog.csdn.net/xiaowei_cqu/article/details/19839019

  时间对比
  通过迭代二十次取平均时间，得到每种方法是运算时间如下：
  using .ptr and [] = 22.0437ms
  using .ptr and * ++ = 20.168ms
  using .ptr and * ++ and modulo = 20.0039ms
  using .ptr and * ++ and bitwise = 10.236ms
  using direct pointer arithmetic = 10.1062ms
  using .ptr and * ++ and bitwise with img.cols * img.channels() = 123.648ms
  using .ptr and * ++ and bitwise <continuous> = 10.4485ms
  using .ptr and * ++ and bitwise <continuous+channels> = 9.28796ms
  using Mat_ iterator = 674.741ms
  using Mat_ iterator and bitwise = 663.554ms
  using MatIterator_ = 911.159ms
  using at = 1143.83ms
  using input/output images = 11.0377ms
  using overloaded operators = 8.69846ms

  指针*++访问和位运算是最快的方法；而不断的计算image.cols*image.channles()花费了大量重复的时间；
  另外迭代器访问虽然安全，但性能远低于指针运算；
  通过图像坐标(j,i)访问时最慢的，使用重载操作符直接运算效率最高。
*/

// 方法零：ptr和[]操作符
// Mat最直接的访问方法是通过.ptr<>函数得到一行的指针，并用[]操作符访问某一列的像素值。
// using .ptr and [] 
void colorReduce0(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			data[i] = data[i] / div*div + div / 2;
		}
	}
}

// 方法一：.ptr和指针操作
// 除了[]操作符，我们可以移动指针*++的组合方法访问某一行中所有像素的值。
// using .ptr and * ++   
void colorReduce1(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			*data++ = *data / div*div + div / 2;
		}
	}
}

// 方法二：ptr、指针操作和取模运算
// 方法二和方法一的访问方式相同，不同的是color reduce用模运算代替整数除法
// using .ptr and * ++ and modulo  
void colorReduce2(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			int v = *data;
			*data++ = v - v%div + div / 2;

		}
	}
}

// 方法三：.ptr、指针运算和位运算
// 由于进行量化的单元div通常是2的整次方，因此所有的乘法和除法都可以用位运算表示。
// using .ptr and * ++ and bitwise  
void colorReduce3(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			*data++ = *data&mask + div / 2;
		}
	}
}

// 方法四：指针运算
// 方法四和方法三量化处理的方法相同，不同的是用指针运算代替*++操作。
// direct pointer arithmetic  
void colorReduce4(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	int step = img.step;// effective width
	// mask used to round the pixel value
	uchar mask = 0xFF << n;// e.g. for div=16, mask=0xF0
	// get the pointer to the img buffer
	uchar *data = img.data;
	for (int j = 0;j < nr;j++)
	{
		for (int i = 0;i < nc;i++)
		{
			*(data + i) = *data&mask + div / 2;
		}
		data += step;// next line
	}
}

// 方法五：.ptr、*++、位运算以及image.cols * image.channels()
// 这种方法就是没有计算nc，基本是个充数的方法。
// using .ptr and * ++ and bitwise with image.cols * image.channels()  
void colorReduce5(Mat &img, int div = 64)
{
	int nr = img.rows;
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < img.cols*img.channels();i++)
		{
			*data++ = *data&mask + div / 2;
		}
	}
}

// 方法六：连续图像
// Mat提供了isContinuous()函数用来查看Mat在内存中是不是连续存储，如果是则图片被存储在一行中。
// using .ptr and * ++ and bitwise (continuous)  
void colorReduce6(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols * img.channels();
	if (img.isContinuous())
	{
		// then no padded pixels
		nc = nc*nr;
		nr = 1; // it is now a 1D array
	}

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			*data++ = *data&mask + div / 2;
		}
	}
}

// 方法七：continuous+channels
// 与方法六基本相同，也是充数的。
// using .ptr and * ++ and bitwise (continuous+channels)  
void colorReduce7(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols;
	if (img.isContinuous())
	{
		nc = nc * nr;
		nr = 1;
	}
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	for (int j = 0;j < nr;j++)
	{
		uchar* data = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++) 
		{
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
		}
	}
}

// 方法八：Mat _iterator
// 真正有区别的方法来啦，用Mat提供的迭代器代替前面的[]操作符或指针，血统纯正的官方方法~
// using Mat_ iterator   
void colorReduce8(Mat &img, int div = 64)
{
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();
	for (;it != itend;++it)
	{
		(*it)[0] = (*it)[0] / div * div + div / 2;
		(*it)[1] = (*it)[1] / div * div + div / 2;
		(*it)[2] = (*it)[2] / div * div + div / 2;
	}
	// 倒着迭代
	/*for (;it != itend;--itend)
	{
		(*itend)[0] = (*itend)[0] / div * div + div / 2;
		(*itend)[1] = (*itend)[1] / div * div + div / 2;
		(*itend)[2] = (*itend)[2] / div * div + div / 2;
	}*/
}

// 方法九：Mat_ iterator 和位运算
// using Mat_ iterator and bitwise  
void colorReduce9(Mat &img, int div = 64)
{
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = img.end<Vec3b>();

	for (;it != itend;++it)
	{
		(*it)[0] = (*it)[0] & mask + div / 2;
		(*it)[1] = (*it)[1] & mask + div / 2;
		(*it)[2] = (*it)[2] & mask + div / 2;
	}
}

// 方法十：MatIterator_
// using MatIterator_   
void colorReduce10(Mat &img, int div = 64)
{
	Mat_<Vec3b> cimg = img;
	Mat_<Vec3b>::iterator it = cimg.begin();
	Mat_<Vec3b>::iterator itend = cimg.end();
	for (;it != itend;it++)
	{
		(*it)[0] = (*it)[0] / div*div + div / 2;
		(*it)[1] = (*it)[1] / div*div + div / 2;
		(*it)[2] = (*it)[2] / div*div + div / 2;
	}
}

// 方法十一：图像坐标
// using (j,i) 
void colorReduce11(Mat &img, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols;
	for (int j = 0;j < nr;j++)
	{
		for (int i = 0;i < nc;i++)
		{
			img.at<Vec3b>(j, i)[0] = img.at<Vec3b>(j, i)[0] / div*div + div / 2;
			img.at<Vec3b>(j, i)[1] = img.at<Vec3b>(j, i)[1] / div*div + div / 2;
			img.at<Vec3b>(j, i)[2] = img.at<Vec3b>(j, i)[2] / div*div + div / 2;
		}
	}
}

// 方法十二：创建输出图像
// with input/ouput images 
void colorReduce12(const Mat &img, Mat &result, int div = 64)
{
	int nr = img.rows;
	int nc = img.cols;

	result.create(nr, nc, img.type());
	
	nc = nc*nr;
	nr = 1;
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	for (int j = 0;j < nr;j++)
	{
		uchar* data = result.ptr<uchar>(j);
		const uchar* idata = img.ptr<uchar>(j);
		for (int i = 0;i < nc;i++)
		{
			// 因为是三通道
			*data++ = (*idata++)&mask + div / 2;
			*data++ = (*idata++)&mask + div / 2;
			*data++ = (*idata++)&mask + div / 2;
		}
	}
}


// 方法十三：重载操作符
// Mat重载了+&等操作符，可以直接将两个Scalar(B,G,R)数据进行位运算和数学运算。
// using overloaded operators  
void colorReduce13(Mat &img, int div = 64)
{
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
	uchar mask = 0xFF << n;
	img = (img&Scalar(mask, mask, mask)) + Scalar(div / 2, div / 2, div / 2);
}

int main()
{
	Mat m = imread("d:\\123.png", 1);
	Mat result;
	imshow("img", m);
	colorReduce0(m);
	imshow("0",m);
	waitKey(0);
	return 0;
}