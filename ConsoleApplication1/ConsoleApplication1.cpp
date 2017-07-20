// Grabcut.cpp : �������̨Ӧ�ó������ڵ㡣  
//  Graph Cut  
// �ο���http://blog.csdn.net/wangyaninglm/article/details/44151213
// �����ǲ��õ�opencv�ٷ�sample
// opencv�汾��2.413   2413
//
//GrabCut����˵��
//void grabCut(InputArray img, InputOutputArray mask, Rect rect, InputOutputArraybgdModel, InputOutputArrayfgdModel, intiterCount, intmode = GC_EVAL)
//Parameters:
//image �C Input 8 - bit 3 - channel image.
//mask �C
//Input / output 8 - bit single - channel mask.The mask is initialized by the function whenmode is set to GC_INIT_WITH_RECT.Its elements may have one of following values :
//GC_BGD defines an obvious background pixels.
//GC_FGD defines an obvious foreground(object) pixel.
//GC_PR_BGD defines a possible background pixel.
//GC_PR_BGD defines a possible foreground pixel.
//rect �C ROI containing a segmented object.The pixels outside of the ROI are marked as ��obvious background��.The parameter is only used whenmode == GC_INIT_WITH_RECT .
//bgdModel �C Temporary array for the background model.Do not modify it while you are processing the same image.
//fgdModel �C Temporary arrays for the foreground model.Do not modify it while you are processing the same image.
//iterCount �C Number of iterations the algorithm should make before returning the result.Note that the result can be refined with further calls withmode == GC_INIT_WITH_MASK ormode == GC_EVAL .
//mode �C
//Operation mode that could be one of the following :
//GC_INIT_WITH_RECT The function initializes the state and the mask using the provided rectangle.After that it runsiterCountiterations of the algorithm.
//GC_INIT_WITH_MASK The function initializes the state using the provided mask.Note thatGC_INIT_WITH_RECT andGC_INIT_WITH_MASK can be combined.Then, all the pixels outside of the ROI are automatically initialized withGC_BGD .
//GC_EVAL The value means that the algorithm should just resume.
//
//����ԭ�ͣ�
//void cv::grabCut(const Mat& img, Mat& mask, Rect rect,
//	Mat& bgdModel, Mat& fgdModel,
//	int iterCount, int mode)
//	���У�
//	img�������ָ��Դͼ�񣬱�����8λ3ͨ����CV_8UC3��ͼ���ڴ���Ĺ����в��ᱻ�޸ģ�
//	mask��������ͼ�����ʹ��������г�ʼ������ômask�����ʼ��������Ϣ����ִ�зָ��ʱ��Ҳ���Խ��û��������趨��ǰ���뱳�����浽mask�У�Ȼ���ٴ���grabCut�������ڴ������֮��mask�лᱣ������maskֻ��ȡ��������ֵ��
//	GCD_BGD�� = 0����������
//	GCD_FGD�� = 1����ǰ����
//	GCD_PR_BGD�� = 2�������ܵı�����
//	GCD_PR_FGD�� = 3�������ܵ�ǰ����
//	���û���ֹ����GCD_BGD����GCD_FGD����ô���ֻ����GCD_PR_BGD��GCD_PR_FGD��
//	rect���������޶���Ҫ���зָ��ͼ��Χ��ֻ�иþ��δ����ڵ�ͼ�񲿷ֲű�����
//	bgdModel��������ģ�ͣ����Ϊnull�������ڲ����Զ�����һ��bgdModel��bgdModel�����ǵ�ͨ�������ͣ�CV_32FC1��ͼ��������ֻ��Ϊ1������ֻ��Ϊ13x5��
//	fgdModel����ǰ��ģ�ͣ����Ϊnull�������ڲ����Զ�����һ��fgdModel��fgdModel�����ǵ�ͨ�������ͣ�CV_32FC1��ͼ��������ֻ��Ϊ1������ֻ��Ϊ13x5��
//	iterCount���������������������0��
//	mode��������ָʾgrabCut��������ʲô��������ѡ��ֵ�У�
//	GC_INIT_WITH_RECT�� = 0�����þ��δ���ʼ��GrabCut��
//	GC_INIT_WITH_MASK�� = 1����������ͼ���ʼ��GrabCut��
//	GC_EVAL�� = 2����ִ�зָ

#include "stdafx.h"  

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  

#include <iostream>  

#include "ComputeTime.h"  
#include "windows.h"  

using namespace std;
using namespace cv;

static void help()
{
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
const Scalar alpha = Scalar(0, 0, 0, 0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;  //Ctrl��  
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY; //Shift��  

static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;  //�õ�mask�����λ,ʵ������ֻ����ȷ���Ļ����п��ܵ�ǰ���㵱��mask  
}

class GCApplication
{
public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;

	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);

	const string* winName;
	const Mat* image;
	Mat mask;
	//Mat mat(480, 640, CV_8UC4);
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
};

/*����ı�����ֵ*/
void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	    
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;    //NOT_SET == 0  
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

/*����ĳ�Ա������ֵ����*/
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

/*��ʾ4���㣬һ�����κ�ͼ�����ݣ���Ϊ����Ĳ���ܶ�ط���Ҫ�õ�������������Ե����ó���*/
void GCApplication::showImage() const
{
	if (image->empty() || winName->empty())
		return;

	Mat res;
	Mat binMask;
	if (!isInitialized)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		image->copyTo(res, binMask);  //�������λ��0����1�����ƣ�ֻ������ǰ���йص�ͼ�񣬱���˵���ܵ�ǰ�������ܵı���  
	}

	vector<Point>::const_iterator it;
	/*����4������ǽ�ѡ�е�4�����ò�ͬ����ɫ��ʾ����*/
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)  //���������Կ�����һ��ָ��  
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)  //ȷ����ǰ���ú�ɫ��ʾ  
		circle(res, *it, radius, RED, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	/*������*/
	if (rectState == IN_PROCESS || rectState == SET)
	{
		Mat ddd;
		//res.copyTo(ddd);
		cvtColor(res, ddd, CV_RGB2RGBA);
		for (int i = 0; i<ddd.rows * ddd.cols; i++)
		{
			//(uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[0] = 0;
			//(uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[1] = 0;
			//(uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[2] = 0;
			if (((uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[0] == 0)
				&& ((uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[1] == 0)
				&& ((uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[2]) == 0)
			{
				(uchar)ddd.at<Vec4b>(i / ddd.cols, i%ddd.cols)[3] = 0;
			}
		}

		imwrite("result-1.png", ddd);
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
	}
		
	imshow(*winName, res);
}

/*�ò�����ɺ�maskͼ����rect�ڲ���3������ȫ��0*/
void GCApplication::setRectInMask()
{
	assert(!mask.empty());
	mask.setTo(GC_BGD);   //GC_BGD == 0  
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));    //GC_PR_FGD == 3�������ڲ�,Ϊ���ܵ�ǰ����  
}

void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr) //ȷ���ĵ�  
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0  
		fvalue = GC_FGD;    //1  
	}
	else    //���ʵ�  
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2  
		fvalue = GC_PR_FGD; //3  
	}
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);   //�õ㴦Ϊ2  
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);   //�õ㴦Ϊ3  
	}
}

/*�����Ӧ����������flagsΪCV_EVENT_FLAG�����*/
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check  
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)//ֻ���������ʱ  
		{
			rectState = IN_PROCESS; //��ʾ���ڻ�����  
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) && rectState == SET) //������alt������shift�����һ����˾��Σ���ʾ���ڻ�ǰ��������  
			lblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET) //���ڻ����ܵ�ǰ��������  
			prLblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));   //���ν���  
			rectState = SET;
			setRectInMask();
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)   //�ѻ���ǰ�󾰵�  
		{
			setLblsInMask(flags, Point(x, y), false);    //����ǰ����  
			lblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true); //����������  
			prLblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();    //���ϵ���ʾͼƬ  
		}
		else if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

void setAlphaMat(Mat &mat)
{
	for (int i = 0;i < mat.rows;++i)
	{
		for (int j = 0;j < mat.cols;++j)
		{
		//	Vec4b&rgba = mat.at<Vec4b>(i, j);
			//rgba[0] = UCHAR_MAX;
			//rgba[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
			//rgba[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
			//rgba[3] = saturate_cast<uchar>(1);
			//rgba[3] = saturate_cast<uchar>(1 * (rgba[1] + rgba[2]));

		}
	}
}


/*�ú�������grabcut�㷨�����ҷ����㷨���е����Ĵ���*/
int GCApplication::nextIter()
{
	if (isInitialized)
	{

		//ʹ��grab�㷨����һ�ε���������2Ϊmask��������maskλ�ǣ������ڲ�������Щ�����Ǳ��������Ѿ�ȷ���Ǳ���������еĵ㣬��maskͬʱҲΪ���  
		//������Ƿָ���ǰ��ͼ��  
		grabCut(*image, mask, rect, bgdModel, fgdModel, 1);
	}
	else
	{
		if (rectState != SET)
			return iterCount;

		if (lblsState == SET || prLblsState == SET)
		{
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		}
		else
		{
			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
		}

		isInitialized = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}

GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}

void Sharpen2(const Mat& myImage, Mat& Result)
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


int main(int argc, char** argv)
{
	string filename;
	cout << " Grabcuts ! \n";
	cout << "input image name:  " << endl;
	cin >> filename;


	Mat ori = imread(filename, 1);

	vector<Mat> rgb_planes;
	split(ori, rgb_planes);
	equalizeHist(rgb_planes[0], rgb_planes[0]);
	equalizeHist(rgb_planes[1], rgb_planes[1]);
	equalizeHist(rgb_planes[2], rgb_planes[2]);

	// ֱ��ͼ�Ż�
	Mat image_hist;
	merge(rgb_planes, image_hist);
	// ��
	Mat image;
	Sharpen2(ori, image);

	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}

	help();

	const string winName = "image";
	cvNamedWindow(winName.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(winName.c_str(), on_mouse, 0);

	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();

	for (;;)
	{
		int c = cvWaitKey(0);
		switch ((char)c)
		{
		case 'b':
			cout << "input image name again:  " << endl;
			cin >> filename;
			image = imread(filename, 1);
			if (image.empty())
			{
				cout << "\n Durn, couldn't read image filename " << filename << endl;
				goto exit_main;
			}

			help();
			gcapp.setImageAndWinName(image, winName);
			gcapp.reset();
			gcapp.showImage();
			break;
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':
			ComputeTime ct;
			ct.Begin();

			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
				cout << "����ʱ��:  " << ct.End() << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}

exit_main:
	cvDestroyWindow(winName.c_str());
	return 0;
}

// ��rgb��ͨ��ת����������ɫͨ�����Ҷȣ�rgba...
// �ο���http://blog.csdn.net/gdfsg/article/details/50927257
int main_channels() {
	Mat image = imread("seg_result2.png");
	Mat imgGRAY, imgRGBA, imgRGB555;
	cvtColor(image, imgGRAY, CV_RGB2GRAY);// ��rgbת���ɻҶ�ͼ��
	cvtColor(image, imgRGBA, CV_RGB2RGBA); // ��rgbת����rgba
	cvtColor(image, imgRGB555, CV_RGB2BGR555);// �������rgb555��ʲô��

	int n = image.channels();
	int nRGBA = imgRGBA.channels();
	int nRGB555 = imgRGB555.channels();
	imshow("image", image);
	imshow("imgGRAY", imgGRAY);
	imshow("imgRGBA", imgRGBA);
	waitKey();
	return 0;
}