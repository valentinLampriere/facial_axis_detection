#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;

struct tLine {
	Point p1;
	Point p2;
	tLine() {
		p1 = Point(0,0);
		p2 = Point(0, 0);
	}

	tLine(Point _p1, Point _p2) {
		p1 = _p1;
		p2 = _p2;
	}
};

Mat image;

bool eyeLineDrawn, noseLineDrawn;
vector<Point> eye_points;

tLine eyeLine, noseLine;

const float pi = atan(1.0) * 4;

// Intersect points :
Point intersect_referenceAxis_lineEye;
Point intersect_detectedAxis_lineEye;
Point intersect_detectedAxis_referenceAxis;

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

double getMeanGrayLevelDifference(Mat img, Point a, Point b, vector<double> gld)
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
                //cout << i << " " << j << " " << sp.x << " " << sp.y << std::endl;
                countd++;
                diff = 255;
            }
            meanDiff += diff;
            //cout << i << " / " << j << " sym : " << sp.x << " / " << sp.y << " diff : " << diff << std::endl;
        }
    }

    meanDiff /= count;
    //cout << "meanDiff : " << meanDiff << std::endl;
    //cout << countd << std::endl;;

    return meanDiff;
}

Mat rotate(Mat src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

Point getCenterLine(tLine l) {
	Point center;
	center.x = (l.p1.x + l.p2.x) / 2;
	center.y = (l.p1.y + l.p2.y) / 2;
	return center;
}

double getLengthLine(tLine l) {
	return sqrt(pow(l.p2.x - l.p1.x, 2) + pow(l.p2.y - l.p1.y, 2));
}

Point rotatePoint(Point p, Point pivot, double angle) {
	return Point((p.x - pivot.x) * cos(angle) - (p.y - pivot.y) * sin(angle) + p.x, (p.x - pivot.x) * sin(angle) + (p.y - pivot.y) * cos(angle) + p.y);
}

double calcTeta(Point A, Point B, Point C) {

	double AB2 = pow(B.x - A.x, 2) + pow(B.y - A.y, 2);
	double AC2 = pow(C.x - A.x, 2) + pow(C.y - A.y, 2);
	double BC2 = pow(C.x - B.x, 2) + pow(C.y - B.y, 2);

	double lengthAB = sqrt(AB2);
	double lengthAC = sqrt(AC2);
	double lengthBC = sqrt(BC2);

	return (acos((AC2 + AB2 - BC2) / (2 * lengthAC * lengthAB))) * 180 / pi;
}
tLine extendsLine(tLine l) {
	float alpha, angle;

	try {
		alpha = (float)((float)noseLine.p2.y - (float)noseLine.p1.y) / ((float)noseLine.p2.x - (float)noseLine.p1.x);
		angle = atan(alpha);
		Point p1 = Point(l.p1.x - 1000 * cos(angle), l.p1.y - 1000 * sin(angle));
		Point p2 = Point(l.p2.x + 1000 * cos(angle), l.p2.y + 1000 * sin(angle));

		return tLine(p1, p2);
	}
	catch (Exception e1) {}
	return l;


}

// From stackoverflow : https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function/7448287#7448287
bool intersection(tLine l1, tLine l2, Point &r) {
	Point x = l2.p1 - l1.p1;
	Point d1 = l1.p2 - l1.p1;
	Point d2 = l2.p2 - l2.p1;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = l1.p1 + d1 * t1;
	return true;
}

// Define the run method
void run();

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
				cv::line(image, eye_points.at(0), eye_points.at(1), Scalar(255, 255, 255), 1);
				eyeLineDrawn = true;
				cv::imshow("Image crop", image);
			}
		}
		// If the two eyes have already been clicked
		else if(!noseLineDrawn) {
			
			float alpha;
			float angle;
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

			/* Rotate image 
			if (alpha < 0) {
				image = rotate(image, angle + 90);
			}
			else {
				image = rotate(image, angle - 90);
			}*/

			noseLineDrawn = true;

			noseLine = extendsLine(noseLine);

			intersection(eyeLine, noseLine, intersect_referenceAxis_lineEye);

			line(image, noseLine.p1, noseLine.p2, Scalar(255, 255, 255), 1);

			run();

			imshow("Image crop", image);	
		}
	}
}

void run() {

	double bestGrayLevel = 255;
	int shiftBestGrayLevel = 0;
	double bestGrayLevelRotate = 255;
	int shiftBestGrayLevelRotate = 0;

	double finalShift, finalTeta;

	Point a = Point(image.size().width / 2, 0); // first point of the axis
	Point b = Point(image.size().width / 2, image.size().height); // second point of the axis

	vector<double> vec;
	int middleX = image.size().width / 2;
	if ((image.size().width / 2) % 2 == 1) {
		middleX = image.size().width / 2 + 1;
	}


	for (int shift = middleX - 30; shift <= middleX + 30; shift += 10) {
		Point _a = Point(a.x = shift, a.y);
		Point _b = Point(b.x = shift, a.y);

		double grayLevel = getMeanGrayLevelDifference(image, _a, _b, vec);
		if (grayLevel < bestGrayLevel) {
			bestGrayLevel = grayLevel;
			shiftBestGrayLevel = shift;
		}
	}

	cout << "Best Gray level : " << bestGrayLevel << " at " << shiftBestGrayLevel << "\n";

	Point midPoint = Point(image.size().width / 2, image.size().height / 2);

	for (int rotateShift = -15; rotateShift <= 15; rotateShift++) {
		Point _a = rotatePoint(Point(shiftBestGrayLevel, a.y), midPoint, tan(rotateShift));
		Point _b = rotatePoint(Point(shiftBestGrayLevel, b.y), midPoint, tan(rotateShift));
		double grayLevel = getMeanGrayLevelDifference(image, _a, _b, vec);

		if (grayLevel < bestGrayLevelRotate) {
			bestGrayLevelRotate = grayLevel;
			shiftBestGrayLevelRotate = rotateShift;
		}
	}

	cout << "Best Gray level rotate : " << bestGrayLevelRotate << " at " << shiftBestGrayLevelRotate << "\n";

	//getMeanGrayLevelDifference(image, a, b, vec);
	//getSymmetricPointOf(30, 230, img, a, b);

	a = rotatePoint(Point(shiftBestGrayLevel, 0), midPoint, tan(shiftBestGrayLevelRotate));
	b = rotatePoint(Point(shiftBestGrayLevel, image.size().height), midPoint, tan(shiftBestGrayLevelRotate));

	tLine detectedAxis = tLine(a, b);

	intersection(eyeLine, detectedAxis, intersect_detectedAxis_lineEye);
	intersection(noseLine, detectedAxis, intersect_detectedAxis_referenceAxis);

	finalShift = getLengthLine(tLine(intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye));
	finalTeta = calcTeta(intersect_detectedAxis_referenceAxis, intersect_referenceAxis_lineEye, intersect_detectedAxis_lineEye);

	cout << "FINAL SHIFT : " << finalShift << "\n";
	cout << "FINAL TETA : " << finalTeta << "\n";
	
	putText(image, to_string(finalShift), getCenterLine(tLine(intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye)), 1, 1, Scalar(0, 0, 255));
	line(image, intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye, Scalar(0, 0, 255), 3);
	line(image, a, b, Scalar(255, 255, 255), 1);

}

int main(void) {

	string path("../images/image_crop.jpg");

	image = imread(path);

	if(image.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
    
    
    imshow("calcHist Demo", histogram(image));
    
	imshow("Image crop", image);

	setMouseCallback("Image crop", onMouse);

    waitKey(0);
    return 0;
}
