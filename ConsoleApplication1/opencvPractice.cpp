#include "stdafx.h"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream>  
#include "windows.h"  

using namespace std;
using namespace cv;

void Sharpen(const Mat& myImage, Mat& Result)
{
	CV_Assert(myImage.depth() == CV_8U);  // ������ucharͼ��

	Result.create(myImage.size(), myImage.type());
	const int nChannels = myImage.channels();

	for (int j = 1; j < myImage.rows - 1; ++j)//�ӵ�һ�п�ʼ��������0;��rows-2�н���
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);//��һ��
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);//��һ��

		uchar* output = Result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * current[i]
				- current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
		}
	}

	//��Ե������Եû�о�������Ա�Ե��ֵΪԭͼ���Ӧ���ص�ֵ��

	//�ȴ����Ϻ���
	uchar* output_top = Result.ptr<uchar>(0);
	const uchar* origin_top = myImage.ptr<uchar>(0);
	uchar* output_bottom = Result.ptr<uchar>(myImage.rows - 1);
	const uchar* origin_bottom = Result.ptr<uchar>(myImage.rows - 1);
	for (int i = 0; i < nChannels * myImage.cols - 1; ++i) {
		*output_top++ = *origin_top++;
		*output_bottom++ = *origin_bottom++;
	}

	//����� �ֱ���nChannel
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

// �ο���http://www.cnblogs.com/korbin/p/5618559.html
int main_opencv(int argc, char* argv[]) {
	const char* path = "2.jpg";
	Mat img = imread(path);
	if (img.empty())
	{
		cout << "error";
		return -1;
	}
	imshow("ԭͼ��", img);

	Mat h_kern = (Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Mat h_mat;
	// ��ͼ����о��
	// ����������������ͼ����о������ͼ��������������Ϊ���ڽ����ضԣ����������������������ص�Ӱ�죻Ӱ���Сȡ���ھ���˶�Ӧλ��ֵ�ô�С��
	filter2D(img, h_mat, img.depth(), h_kern);
	imshow("ˮƽ�����Ե��ȡ", h_mat);

	Mat v_kern = (Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
	Mat v_mat;
	filter2D(img, v_mat, img.depth(), v_kern);

	imshow("���ԷǾ�ֵ�˲�2", v_mat);

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

	// ����ֱ��ͼ
	//��BGRͼ��ָ�Ϊ��ͨ��ͼ��
	vector<Mat> bgr_planes;

	split(img, bgr_planes);
	cout << bgr_planes.size() << endl;
	//����ֱ��ͼ
	vector<Mat> hist_image;
	hist_image.resize(3);

	//ֱ��ͼͳ�����������
	const int histSize[] = { 255 };
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	const int channels[] = { 0 };

	calcHist(&bgr_planes[0], 1, channels, Mat(), hist_image[0], 1, histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, channels, Mat(), hist_image[1], 1, histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, channels, Mat(), hist_image[2], 1, histSize, &histRange, uniform, accumulate);
	/// ��ֱ��ͼ�߶ȹ�һ������Χ [ 0, histImage.rows ]
	normalize(hist_image[0], hist_image[0], 0, hist_image[0].rows, NORM_MINMAX, -1, Mat());
	normalize(hist_image[1], hist_image[1], 0, hist_image[1].rows, NORM_MINMAX, -1, Mat());
	normalize(hist_image[2], hist_image[2], 0, hist_image[2].rows, NORM_MINMAX, -1, Mat());

	// ����ֱ��ͼ����  
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize[0]);//ÿ�����ؿ��

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// ��ֱ��ͼ�����ϻ���ֱ��ͼ��Mat����ϵ��ԭ�������Ͻǣ�ϰ���õ�����ϵԭ�������½ǣ���˸߶�Ҫ������������height - y
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

	/// ��ʾֱ��ͼ
	imshow("Hist", histImage);

	// ֱ��ͼ���⻯����������ͼ��Աȶȵķ���������˼���ǰ�ĳЩ����ֵ���е�������䵽��������ֵ�ϣ���ֱ��ͼ�ϵı��־���������ֱ��ͼ������ֱ��ͼ�������ͼ�������ȣ�δ�ı䣬ֻ�����·����˶��ѡ�
	// ����ֱ��ͼ���⻯��ͼ������ؼ������٣�һЩͼ��ϸ�ڿ�����ʧ�������
	// ����ͨ��ֱ��ͼ���⻯  
	vector<Mat> rgb_planes;
	split(img, rgb_planes);
	equalizeHist(rgb_planes[0], rgb_planes[0]);
	equalizeHist(rgb_planes[1], rgb_planes[1]);
	equalizeHist(rgb_planes[2], rgb_planes[2]);

	Mat new_img;
	merge(rgb_planes, new_img);
	imshow("ֱ��ͼ���⻯", new_img);

	// �ɻ�
	Mat result;
	Sharpen(img, result);
	imshow("�ɻ�", result);

	//��ֵ�˲�
	Mat blur_mat;
	blur(img, blur_mat, Size(3, 3));
	imshow("��ֵ�˲�", blur_mat);

	//���ԷǾ�ֵ�˲�
	Mat kern = (Mat_<float>(3, 3) << 1, 2, 1,
		2, 4, 2,
		1, 2, 1) / 16;
	Mat blur2_mat;
	filter2D(img, blur2_mat, img.depth(), kern);
	imshow("���ԷǾ�ֵ�˲�", blur2_mat);

	//��ֵ�˲�
	Mat memedian_mat;
	medianBlur(img, memedian_mat, 3);
	imshow("��ֵ�˲�", memedian_mat);

	//��˹�˲�
	Mat Gaussian_mat;
	GaussianBlur(img, Gaussian_mat, Size(3, 3), 0, 0);
	imshow("��˹�˲�", Gaussian_mat);

	//˫���˲�

	Mat bilateral_mat;
	bilateralFilter(img, bilateral_mat, 25, 25 * 2, 25 / 2);
	imshow("˫���˲�", bilateral_mat);


	waitKey();
	return 0;
}

// ħ������ floodfill
// �ο���http://blog.csdn.net/xiaowei_cqu/article/details/8987387
// �ο���http://blog.csdn.net/poem_qianmo/article/details/28261997

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

// ȫ�ֱ�������
Mat image0, image, gray, mask;// ����ԭʼͼ��Ŀ��ͼ���Ҷ�ͼ����ģͼ 
int ffillMode = 1;//��ˮ����ģʽ
int loDiff = 20, upDiff = 20;//�������ֵ���������ֵ
int connectivity = 4;//��ʾfloodFill������ʶ���Ͱ�λ����ֵͨ  
int isColor = true;//�Ƿ�Ϊ��ɫͼ�ı�ʶ������ֵ  
bool useMask = false;//�Ƿ���ʾ��Ĥ���ڵĲ���ֵ  
int newMaskVal = 255;//�µ����»��Ƶ�����ֵ  

//      �����������ϢonMouse�ص�����  
static void onMouse(int event, int x, int y, int, void*)
{
	// ��������û�а��£��㷵��
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	//-------------------��<1>����floodFill����֮ǰ�Ĳ���׼�����֡�---------------  
	Point seed = Point(x, y);
	int lo = ffillMode == 0 ? 0 : loDiff;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nLowDifference  
	int up = ffillMode == 0 ? 0 : upDiff;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nUpDifference
	int flags = connectivity + (newMaskVal << 8) +
		//��ʶ����0~7λΪg_nConnectivity��8~15λΪg_nNewMaskVal����8λ��ֵ��16~23λΪCV_FLOODFILL_FIXED_RANGE����0��
		(ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	//�������bgrֵ
	int b = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	int g = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	int r = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	
	Rect ccomp;//�����ػ��������С�߽��������  
    
    //���ػ��������ص���ֵ�����ǲ�ɫͼģʽ��ȡScalar(b, g, r)�����ǻҶ�ͼģʽ��ȡScalar(r*0.299 + g*0.587 + b*0.114)  
	Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	newVal = Scalar(255, 255, 255);
	Mat dst = isColor ? image : gray;//Ŀ��ͼ�ĸ�ֵ 
	int area;

	//--------------------��<2>��ʽ����floodFill������----------------------------- 
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

//-----------------------------------��main( )������--------------------------------------------    
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ    
//-----------------------------------------------------------------------------------------------    
int main_floodfill()
{
	//�ı�console������ɫ    
	//system("color 2F");

	//����ԭͼ  
	char* filename = "10.jpg";
	image0 = imread(filename, 1);
	// image0 = imread(filename, -1);// imread��һ������Ϊ-1��ʾ��ȡ����alphaͨ����ͼƬ

	if (image0.empty())
	{
		cout << "Image empty. Usage: ffilldemo <image_name>\n";
		return 0;
	}
	//��ʾ�������� 
	floodfill_help();

	image0.copyTo(image);//����Դͼ��Ŀ��ͼ
	cvtColor(image0, gray, CV_BGR2GRAY);//ת����ͨ����image0���Ҷ�ͼ  
	mask.create(image0.rows + 2, image0.cols + 2, CV_8UC1);//����image0�ĳߴ�����ʼ����Ĥmask  

	namedWindow("image", 0);// Ч��ͼ
	// namedWindow( "Ч��ͼ",CV_WINDOW_AUTOSIZE );  

	//����Trackbar  
	createTrackbar("lo_diff", "image", &loDiff, 255, 0);// �������ֵ
	createTrackbar("up_diff", "image", &upDiff, 255, 0);// �������ֵ

    //���ص�����
	setMouseCallback("image", onMouse, 0);

	//ѭ����ѯ����
	for (;;)
	{
		//����ʾЧ��ͼ 
		imshow("image", isColor ? image : gray);

		int c = waitKey(0);
		//�ж�ESC�Ƿ��£������±��˳�
		if ((c & 255) == 27)
		{
			cout << "Exiting ...\n";
			break;
		}
		//���ݰ����Ĳ�ͬ�����и��ֲ��� 
		switch ((char)c)
		{
		case 'c':
			if (isColor)
			{
				cout << "Grayscale mode is set\n";// �л���ɫͼ/�Ҷ�ͼģʽ
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
			// ��ʾ/������Ĥ����
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
			cout << "Original image is restored\n";// �ָ�ԭʼͼ��
			image0.copyTo(image);
			cvtColor(image, gray, CV_BGR2GRAY);
			mask = Scalar::all(0);
			break;
		case 's':
			cout << "Simple floodfill mode is set\n";// ʹ�ÿշ�Χ����ˮ���
			ffillMode = 0;
			break;
		case 'f':
			cout << "Fixed Range floodfill mode is set\n";// ʹ�ý��䡢�̶���Χ����ˮ���
			ffillMode = 1;
			break;
		case 'g':
			cout << "Gradient (floating range) floodfill mode is set\n";//  ʹ�ý��䡢������Χ����ˮ���
			ffillMode = 2;
			break;
		case '4':
			cout << "4-connectivity mode is set\n";// ������ʶ���ĵͰ�λʹ��4λ������ģʽ
			connectivity = 4;
			break;
		case '8':
			cout << "8-connectivity mode is set\n";// ������־���ĵͰ�λʹ��8λ������ģʽ
			connectivity = 8;
			break;
		}
	}

	return 0;
}

