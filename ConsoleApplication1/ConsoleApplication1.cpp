// Grabcut.cpp : 定义控制台应用程序的入口点。  
//  Graph Cut  
// 参考：http://blog.csdn.net/wangyaninglm/article/details/44151213
// 代码是采用的opencv官方sample
// opencv版本是2.413   2413
//
//GrabCut函数说明
//void grabCut(InputArray img, InputOutputArray mask, Rect rect, InputOutputArraybgdModel, InputOutputArrayfgdModel, intiterCount, intmode = GC_EVAL)
//Parameters:
//image – Input 8 - bit 3 - channel image.
//mask –
//Input / output 8 - bit single - channel mask.The mask is initialized by the function whenmode is set to GC_INIT_WITH_RECT.Its elements may have one of following values :
//GC_BGD defines an obvious background pixels.
//GC_FGD defines an obvious foreground(object) pixel.
//GC_PR_BGD defines a possible background pixel.
//GC_PR_BGD defines a possible foreground pixel.
//rect – ROI containing a segmented object.The pixels outside of the ROI are marked as “obvious background”.The parameter is only used whenmode == GC_INIT_WITH_RECT .
//bgdModel – Temporary array for the background model.Do not modify it while you are processing the same image.
//fgdModel – Temporary arrays for the foreground model.Do not modify it while you are processing the same image.
//iterCount – Number of iterations the algorithm should make before returning the result.Note that the result can be refined with further calls withmode == GC_INIT_WITH_MASK ormode == GC_EVAL .
//mode –
//Operation mode that could be one of the following :
//GC_INIT_WITH_RECT The function initializes the state and the mask using the provided rectangle.After that it runsiterCountiterations of the algorithm.
//GC_INIT_WITH_MASK The function initializes the state using the provided mask.Note thatGC_INIT_WITH_RECT andGC_INIT_WITH_MASK can be combined.Then, all the pixels outside of the ROI are automatically initialized withGC_BGD .
//GC_EVAL The value means that the algorithm should just resume.
//
//函数原型：
//void cv::grabCut(const Mat& img, Mat& mask, Rect rect,
//	Mat& bgdModel, Mat& fgdModel,
//	int iterCount, int mode)
//	其中：
//	img——待分割的源图像，必须是8位3通道（CV_8UC3）图像，在处理的过程中不会被修改；
//	mask——掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割的时候，也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函数；在处理结束之后，mask中会保存结果。mask只能取以下四种值：
//	GCD_BGD（ = 0），背景；
//	GCD_FGD（ = 1），前景；
//	GCD_PR_BGD（ = 2），可能的背景；
//	GCD_PR_FGD（ = 3），可能的前景。
//	如果没有手工标记GCD_BGD或者GCD_FGD，那么结果只会有GCD_PR_BGD或GCD_PR_FGD；
//	rect——用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
//	bgdModel——背景模型，如果为null，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；
//	fgdModel——前景模型，如果为null，函数内部会自动创建一个fgdModel；fgdModel必须是单通道浮点型（CV_32FC1）图像，且行数只能为1，列数只能为13x5；
//	iterCount——迭代次数，必须大于0；
//	mode——用于指示grabCut函数进行什么操作，可选的值有：
//	GC_INIT_WITH_RECT（ = 0），用矩形窗初始化GrabCut；
//	GC_INIT_WITH_MASK（ = 1），用掩码图像初始化GrabCut；
//	GC_EVAL（ = 2），执行分割。

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

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;  //Ctrl键  
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY; //Shift键  

static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;  //得到mask的最低位,实际上是只保留确定的或者有可能的前景点当做mask  
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

/*给类的变量赋值*/
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

/*给类的成员变量赋值而已*/
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

/*显示4个点，一个矩形和图像内容，因为后面的步骤很多地方都要用到这个函数，所以单独拿出来*/
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
		image->copyTo(res, binMask);  //按照最低位是0还是1来复制，只保留跟前景有关的图像，比如说可能的前景，可能的背景  
	}

	vector<Point>::const_iterator it;
	/*下面4句代码是将选中的4个点用不同的颜色显示出来*/
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)  //迭代器可以看成是一个指针  
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)  //确定的前景用红色表示  
		circle(res, *it, radius, RED, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	/*画矩形*/
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

/*该步骤完成后，mask图像中rect内部是3，外面全是0*/
void GCApplication::setRectInMask()
{
	assert(!mask.empty());
	mask.setTo(GC_BGD);   //GC_BGD == 0  
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));    //GC_PR_FGD == 3，矩形内部,为可能的前景点  
}

void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr) //确定的点  
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0  
		fvalue = GC_FGD;    //1  
	}
	else    //概率点  
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2  
		fvalue = GC_PR_FGD; //3  
	}
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);   //该点处为2  
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);   //该点处为3  
	}
}

/*鼠标响应函数，参数flags为CV_EVENT_FLAG的组合*/
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check  
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)//只有左键按下时  
		{
			rectState = IN_PROCESS; //表示正在画矩形  
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) && rectState == SET) //按下了alt键或者shift键，且画好了矩形，表示正在画前景背景点  
			lblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels  
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET) //正在画可能的前景背景点  
			prLblsState = IN_PROCESS;
	}
	break;
	case CV_EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));   //矩形结束  
			rectState = SET;
			setRectInMask();
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)   //已画了前后景点  
		{
			setLblsInMask(flags, Point(x, y), false);    //画出前景点  
			lblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true); //画出背景点  
			prLblsState = SET;
			showImage();
		}
		break;
	case CV_EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();    //不断的显示图片  
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


/*该函数进行grabcut算法，并且返回算法运行迭代的次数*/
int GCApplication::nextIter()
{
	if (isInitialized)
	{

		//使用grab算法进行一次迭代，参数2为mask，里面存的mask位是：矩形内部除掉那些可能是背景或者已经确定是背景后的所有的点，且mask同时也为输出  
		//保存的是分割后的前景图像  
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

	// 直方图优化
	Mat image_hist;
	merge(rgb_planes, image_hist);
	// 锐化
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
				cout << "运行时间:  " << ct.End() << endl;
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

// 将rgb三通道转换成其他颜色通道，灰度，rgba...
// 参考：http://blog.csdn.net/gdfsg/article/details/50927257
int main_channels() {
	Mat image = imread("seg_result2.png");
	Mat imgGRAY, imgRGBA, imgRGB555;
	cvtColor(image, imgGRAY, CV_RGB2GRAY);// 将rgb转换成灰度图像
	cvtColor(image, imgRGBA, CV_RGB2RGBA); // 将rgb转换成rgba
	cvtColor(image, imgRGB555, CV_RGB2BGR555);// 不懂这个rgb555是什么鬼

	int n = image.channels();
	int nRGBA = imgRGBA.channels();
	int nRGB555 = imgRGB555.channels();
	imshow("image", image);
	imshow("imgGRAY", imgGRAY);
	imshow("imgRGBA", imgRGBA);
	waitKey();
	return 0;
}