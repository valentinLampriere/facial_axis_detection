#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;



int main(void) {

	string path("../images/image_crop.jpg");

	Mat img = imread(path);

	if(img.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    int histSize = 256;

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };

    bool uniform = true, accumulate = false;

    Mat g_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        /*line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);*/
        line(histImage, Point(bin_w * (i), hist_h),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(255, 0, 50), 2, 8, 0);    
    }

    imshow("Source image", img);
    imshow("calcHist Demo", histImage);

    waitKey(0);

    return 0;
}
