#include "stdafx.h"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream>  
#include "windows.h"  

using namespace std;
using namespace cv;

/*
 ��OpenCV������Mat��ÿ�����ص�ֵ���£�
  �ο���http://blog.csdn.net/xiaowei_cqu/article/details/19839019

  ʱ��Ա�
  ͨ��������ʮ��ȡƽ��ʱ�䣬�õ�ÿ�ַ���������ʱ�����£�
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

  ָ��*++���ʺ�λ���������ķ����������ϵļ���image.cols*image.channles()�����˴����ظ���ʱ�䣻
  ���������������Ȼ��ȫ��������Զ����ָ�����㣻
  ͨ��ͼ������(j,i)����ʱ�����ģ�ʹ�����ز�����ֱ������Ч����ߡ�
*/

// �����㣺ptr��[]������
// Mat��ֱ�ӵķ��ʷ�����ͨ��.ptr<>�����õ�һ�е�ָ�룬����[]����������ĳһ�е�����ֵ��
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

// ����һ��.ptr��ָ�����
// ����[]�����������ǿ����ƶ�ָ��*++����Ϸ�������ĳһ�����������ص�ֵ��
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

// ��������ptr��ָ�������ȡģ����
// �������ͷ���һ�ķ��ʷ�ʽ��ͬ����ͬ����color reduce��ģ���������������
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

// ��������.ptr��ָ�������λ����
// ���ڽ��������ĵ�Ԫdivͨ����2�����η���������еĳ˷��ͳ�����������λ�����ʾ��
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

// �����ģ�ָ������
// �����ĺͷ�������������ķ�����ͬ����ͬ������ָ���������*++������
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

// �����壺.ptr��*++��λ�����Լ�image.cols * image.channels()
// ���ַ�������û�м���nc�������Ǹ������ķ�����
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

// ������������ͼ��
// Mat�ṩ��isContinuous()���������鿴Mat���ڴ����ǲ��������洢���������ͼƬ���洢��һ���С�
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

// �����ߣ�continuous+channels
// �뷽����������ͬ��Ҳ�ǳ����ġ�
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

// �����ˣ�Mat _iterator
// ����������ķ�����������Mat�ṩ�ĵ���������ǰ���[]��������ָ�룬Ѫͳ�����Ĺٷ�����~
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
	// ���ŵ���
	/*for (;it != itend;--itend)
	{
		(*itend)[0] = (*itend)[0] / div * div + div / 2;
		(*itend)[1] = (*itend)[1] / div * div + div / 2;
		(*itend)[2] = (*itend)[2] / div * div + div / 2;
	}*/
}

// �����ţ�Mat_ iterator ��λ����
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

// ����ʮ��MatIterator_
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

// ����ʮһ��ͼ������
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

// ����ʮ�����������ͼ��
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
			// ��Ϊ����ͨ��
			*data++ = (*idata++)&mask + div / 2;
			*data++ = (*idata++)&mask + div / 2;
			*data++ = (*idata++)&mask + div / 2;
		}
	}
}


// ����ʮ�������ز�����
// Mat������+&�Ȳ�����������ֱ�ӽ�����Scalar(B,G,R)���ݽ���λ�������ѧ���㡣
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