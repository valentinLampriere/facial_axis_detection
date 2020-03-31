#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat histogram(Mat img)
{
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

    return histImage;
}


Point getSymmetricPointOf(int px, int py, Mat img, Point a, Point b)
{
    Point p = Point(px, py);
    //circle(img, p, 2, Scalar(255, 0, 0));

    Point u = Point(b.x - a.x, b.y - a.y); // direction vector of the axis
    //cout << "u(x) : " << u.x << "u(y) : " << u.y << std::endl;

    double a11, a12, a21, a22, c1, c2, x, y, m, k, c;

    if ((b.x - a.x) != 0) // if the axis is not perfectly vertical
    {
        // equation of the axis line. Let d : mx + ky + c = 0
        m = (b.y - a.y) / (b.x - a.x);
        k = -1; // cause we switch side in this equation
        c = a.y - (m * a.x);

        // Let V(PP').u : a11*x + a12*y + c1 = 0
        a11 = u.x; // u(x)
        a12 = u.y; // u(y)
        c1 = (-p.x * u.x) + (-p.y * u.y); // (-p(x) * u(x)) + (-p(y) * u(y))

        // Let a21*x + a22*y + c2 = 0
        a21 = m; // m
        a22 = k; // b
        c2 = m * p.x + k * p.y + 2 * c; // m*p(x) + k*p(y) + 2*c

        // Cramer method
        // ((c1 * a22) - (c2 * a12)) / ((a11 * a22) - (a12 * a21)) = x 
        //((c2 * a11) - (c1 * a21)) / ((a11 * a22) - (a12 * a21)) = y
        x = ((c1 * a22) - (c2 * a12)) / ((a11 * a22) - (a12 * a21));
        y = ((c2 * a11) - (c1 * a21)) / ((a11 * a22) - (a12 * a21));

        x = -1 * x;
        y = -1 * y;
    }
    else // if it's perfectly vertical
    {
        x = a.x + (a.x - p.x); // x is x' axis value + the distance between p.x and x' axis value
        y = p.y; // symmetric point has the same y
    }

    Point pp = Point(x, y);
    //circle(img, pp, 2, Scalar(255, 0, 0));
   // cout << x << " " << y << std::endl;

    return pp;
}

double getMeanGrayLevelDifference(Mat img, Point a, Point b)
{
    int h = img.size().height;
    int w = img.size().width;
    double count = 0;
    double countd = 0;
    double meanDiff = 0;

    for (int i = 5; i < w-5; i++)
    {
        for (int j = 5; j < h-5; j++)
        {
            count++;

            Point sp = getSymmetricPointOf(i, j, img, a, b);
            double diff;
           // cout << sp.x << sp.x << std::endl;
            if (sp.x >= 0 && sp.x < w && sp.y >= 0 && sp.y < h)
            {
                diff = abs(img.at<uchar>(i, j) - img.at<uchar>(sp.x, sp.y));
            }
            else
            {
                cout << i << " " << j << " " << sp.x << " " << sp.y << std::endl;
                countd++;
                diff = 255;
            }
            meanDiff += diff;
            //cout << i << " / " << j << " sym : " << sp.x << " / " << sp.y << " diff : " << diff << std::endl;
        }
    }

    meanDiff /= count;
    cout << "meanDiff : " << meanDiff << std::endl;
    cout << countd << std::endl;;

    return meanDiff;
}

int main(void) {

	string path("../images/image_crop.jpg");

	Mat img = imread(path);

	if(img.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
    
    Point a = Point(img.size().width / 2, 0); // first point of the axis
    Point b = Point(img.size().width / 2 + 15, img.size().height); // second point of the axis
    line(img, a, b, Scalar(255, 255, 255), 2, 8, 0);

    /*LineIterator it(img, a, b, 8);

    for (int i = 0; i < it.count; i++, ++it)
    {
        Point pt = it.pos();
    }*/

    getMeanGrayLevelDifference(img, a, b);
   // getSymmetricPointOf(30, 230, img, a, b);
    

    imshow("Source image", img);
    imshow("calcHist Demo", histogram(img));

    waitKey(0);

    return 0;
}
