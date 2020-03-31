#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;

struct Line {
	Point p1;
	Point p2;
	Line() {
		p1 = Point(0,0);
		p2 = Point(0, 0);
	}

	Line(Point _p1, Point _p2) {
		p1 = _p1;
		p2 = _p2;
	}
};

Mat image;

bool eyeLineDrawn, noseLineDrawn;
vector<Point> eye_points;

Line eyeLine, noseLine;

Mat histogram(int gld[], int max)
{

    int histSize = 511;
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    double bin_h = (hist_h * 1.0) / max;

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i), hist_h),
            Point(bin_w * (i), hist_h - cvRound(bin_h * gld[i])),
            Scalar(255, 0, 0), 1.5, 8, 0);
    }

    return histImage;
}


Point getSymmetricPointOf(int px, int py, Mat img, Point a, Point b)
{
    Point p = Point(px, py);
   // circle(img, p, 2, Scalar(255, 0, 0));

    Point u = Point(b.x - a.x, b.y - a.y); // direction vector of the axis
    //cout << "u(x) : " << u.x << "u(y) : " << u.y << std::endl;

    double a11, a12, a21, a22, c2, m, k, c;
    int c1, x, y;

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
   // circle(img, pp, 2, Scalar(255, 0, 0));
   // cout << "px : " << x << " py : " << y << std::endl;

    return pp;
}

void getMeanGrayLevelDifference(Mat img, Point a, Point b, int* gld, int &max, vector<double> &differences)
{
    int h = img.rows;
    int w = img.cols;
    double countd = 0;

    for (int i = 5; i < h-5; i++)
    {
        for (int j = 5; j < w-5; j++)
        {
                Point sp = getSymmetricPointOf(j, i, img, a, b);
                int diff;
                int d = 0;
                bool flag = false;

                if (sp.x >= 0 && sp.x < w && sp.y >= 0 && sp.y < h)
                {
                    int p = img.at<uchar>(i, j);
                    //cout << sp.x << " " << sp.y << std::endl;
                    int k = img.at<uchar>(sp.y, sp.x);
                    d = p - k;
                    diff = abs(d);
                    flag = true;

                }
                else
                {
                    countd++; // number of pixels skipped because its symmetric is out of the image due to axis rotation
                    d = img.at<uchar>(i, j) - 0;
                    diff = d;
                    diff = 127;
                }

                if (sp.x != j || sp.y != i) // not a pixel of the line
                {
                    differences.push_back(diff);

                    if (flag)
                    {
                        int k = d + 255; // d is between -255 and 255, so to store it in array of 511 values we do +255
                        gld[k] += 1;


                        if (gld[k] > max)
                        {
                            max = gld[k];
                        }
                    }
                }

                //cout << i << " / " << j << " sym : " << sp.x << " / " << sp.y << " diff : " << diff << std::endl;
        }
    }

    cout << "countd : " << countd << std::endl;;
}

int getMEAN(Mat img, Point a, Point b, double meanDiff)
{
    int h = img.rows;
    int w = img.cols;
    int MEAN = 0;

    for (int i = 5; i < h - 5; i++)
    {
        for (int j = 5; j < w - 5; j++)
        {
                Point sp = getSymmetricPointOf(j, i, img, a, b);
                int diff;

                // cout << sp.x << sp.x << std::endl;
                if (sp.x >= 0 && sp.x < w && sp.y >= 0 && sp.y < h)
                {
                    diff = abs(img.at<uchar>(i, j) - img.at<uchar>(sp.y, sp.x));

                    if (diff >= meanDiff - 4 && diff <= meanDiff + 4)
                    {
                        MEAN++;
                    }
                }
        }
    }

    cout << "MEAN : " << MEAN << std::endl;

    return MEAN;
}

vector<int> iterateLine(Point a, Point b) {
    
    vector<int> linePixels;
	LineIterator it(image, a, b);
	for (int i = 0; i < it.count; i++, ++it)
	{
		Point pt = it.pos();
        linePixels.push_back(pt.x); // get x position of the line pixel at the y row of the image
	}

    return linePixels;
}

Mat rotate(Mat src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

Point getCenterLine(Line l) {
	Point center;
	center.x = (l.p1.x + l.p2.x) / 2;
	center.y = (l.p1.y + l.p2.y) / 2;
	return center;
}

Point rotatePoint(Point p, Point pivot, double angle) {
	return Point((p.x - pivot.x) * cos(angle) - (p.y - pivot.y) * sin(angle) + p.x, (p.x - pivot.x) * sin(angle) + (p.y - pivot.y) * cos(angle) + p.y);
}

Line extendsLine(Line l) {
	float alpha, angle;

	try {
		alpha = (float)((float)noseLine.p2.y - (float)noseLine.p1.y) / ((float)noseLine.p2.x - (float)noseLine.p1.x);
		angle = atan(alpha);
		Point p1 = Point(l.p1.x - 1000 * cos(angle), l.p1.y - 1000 * sin(angle));
		Point p2 = Point(l.p2.x + 1000 * cos(angle), l.p2.y + 1000 * sin(angle));

		return Line(p1, p2);
	}
	catch (Exception e1) {}
	return l;


}

static void onMouse(int event, int x, int y, int, void*) {
	if(event == 1) { // event click press
		// If the two points for the line between eyes haven't been drawn
		if (!eyeLineDrawn) {
			Point p = Point(x, y);
			if (eye_points.size() == 0) {
				eyeLine.p1 = Point(x, y);
				eye_points.push_back(Point(x, y));
			}
			else if (eye_points.size() == 1) {
				eyeLine.p2 = Point(x, y);
				eye_points.push_back(Point(x, y));
				line(image, eye_points.at(0), eye_points.at(1), Scalar(255, 255, 255), 2);
				eyeLineDrawn = true;
				cv::imshow("Image crop", image);
			}
		}
		else if(!noseLineDrawn) {
			
			float alpha;
			float angle;
			float pi = atan(1.0) * 4;
			noseLine.p1 = getCenterLine(eyeLine);
			noseLine.p2 = Point(x, y);
			Point centerNoseLine = getCenterLine(noseLine);

			try {
				alpha = (float)((float)noseLine.p2.y - (float)noseLine.p1.y) / ((float)noseLine.p2.x - (float)noseLine.p1.x);
				angle = atan(alpha) * 180/pi;
			}
			catch (Exception e1) {
				alpha = 0.0;
				angle = 0;
			}
			
			cout << alpha;
			cout << "\n";

			/* Rotate image 
			if (alpha < 0) {
				image = rotate(image, angle + 90);
			}
			else {
				image = rotate(image, angle - 90);
			}*/

			noseLineDrawn = true;

			noseLine = extendsLine(noseLine);

			line(image, noseLine.p1, noseLine.p2, Scalar(255, 255, 255), 2);

			imshow("Image crop", image);	
		}
	}
}

int main(void) {

	string path("../images/eminem.jpg");

	image = imread(path);

	if(image.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
    
    Point a = Point(image.cols / 2 - 300, 0); // first point of the axis
    Point b = Point(image.cols / 2 - 10, image.rows); // second point of the axis
    line(image, a, b, Scalar(255, 255, 255), 1, 8, 0);
 
    //getSymmetricPointOf(100, 500, image, a, b);
    vector<double> differences;

    int gld[511] = { 0 };
    int max = 0;
    getMeanGrayLevelDifference(image, a, b, gld, max, differences);

    cout << "max : " << max << std::endl;

    Scalar meanDiff;
    Scalar std;
    meanStdDev(differences, meanDiff, std);
    double variance = pow(std[0], 2.0);

    cout << "mean : " << meanDiff[0] << std::endl;
    cout << "std : " << std[0] << std::endl;
    cout << "var : " << variance << std::endl;

    int MEAN = getMEAN(image, a, b, meanDiff[0]);
    double score = MEAN / variance;

    cout << "score : " << score << std::endl;

    imshow("calcHist Demo", histogram(gld, max)); 
    
	cv::imshow("Image crop", image);

	setMouseCallback("Image crop", onMouse);

    waitKey(0);
    return 0;
}
