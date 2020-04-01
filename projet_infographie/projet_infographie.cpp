#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <cmath>
#include <sstream>
#include <iomanip>

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

tLine eyeLine, referenceAxis, detectedAxis;

const float pi = atan(1.0) * 4;

// Intersect points :
Point intersect_referenceAxis_lineEye;
Point intersect_detectedAxis_lineEye;
Point intersect_detectedAxis_referenceAxis;

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
	double x, y, xP, yP;
	angle = angle * pi / 180;
	xP = p.x - pivot.x;
	yP = p.y - pivot.y;
	x = xP * cos(angle) + yP * sin(angle) + pivot.x;
	y = - xP * sin(angle) + yP * cos(angle) + pivot.y;
	return Point(x, y);
}

double getAngle(tLine l) {
	return (double)((double)atan2(l.p1.y - l.p2.y, l.p1.x - l.p2.x)) * 180 / pi;
}

double calcTeta(Point A, Point B, Point C) {

	/*double AB2 = pow(B.x - A.x, 2) + pow(B.y - A.y, 2);
	double AC2 = pow(C.x - A.x, 2) + pow(C.y - A.y, 2);
	double BC2 = pow(C.x - B.x, 2) + pow(C.y - B.y, 2);

	double lengthAB = sqrt(AB2);
	double lengthAC = sqrt(AC2);
	double lengthBC = sqrt(BC2);

	return (acos((AC2 + AB2 - BC2) / (2 * lengthAC * lengthAB))) * 180 / pi;
	*/
	return getAngle(tLine(A, B)) - getAngle(tLine(A, C));
}
tLine extendsLine(tLine l) {
	float alpha, angle;

	try {
		alpha = (float)((float)referenceAxis.p2.y - (float)referenceAxis.p1.y) / ((float)referenceAxis.p2.x - (float)referenceAxis.p1.x);
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

void drawDottedLine(Mat img, tLine l, Scalar color) {
	vector<Mat> imgRVB;
	LineIterator it(img, l.p1, l.p2);
	split(img, imgRVB);
	for (int i = 0; i < it.count; i++, it++) {
		Point p = it.pos();
		if (i % 5 > 1) {
			imgRVB[0].at<uchar>(p.y, p.x) = color[0];
			imgRVB[1].at<uchar>(p.y, p.x) = color[1];
			imgRVB[2].at<uchar>(p.y, p.x) = color[2];
		}
	}
	merge(imgRVB, img);
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

				drawDottedLine(image, tLine(eye_points.at(0), eye_points.at(1)), Scalar(255,255,255));

				//line(image, eye_points.at(0), eye_points.at(1), Scalar(255, 255, 255), 1);
				eyeLineDrawn = true;
				imshow("Image crop", image);
			}
		}
		// If the two eyes have already been clicked
		else if(!noseLineDrawn) {
			
			float alpha;
			float angle;
			referenceAxis.p1 = getCenterLine(eyeLine);
			referenceAxis.p2 = Point(x, y);
			Point centerNoseLine = getCenterLine(referenceAxis);

			try {
				alpha = (float)((float)referenceAxis.p2.y - (float)referenceAxis.p1.y) / ((float)referenceAxis.p2.x - (float)referenceAxis.p1.x);
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

			referenceAxis = extendsLine(referenceAxis);

			intersection(eyeLine, referenceAxis, intersect_referenceAxis_lineEye);

			drawDottedLine(image, referenceAxis, Scalar(255, 255, 255));
			//line(image, referenceAxis.p1, referenceAxis.p2, Scalar(255, 255, 255), 1);

			run();

			imshow("Image crop", image);	
		}
	}
}

void run() {

	Mat imgCopy;

	double bestScore = 0;
	int shiftBestScore = 0;
	double bestScoreRotate = 0;
	int shiftBestScoreRotate = 0;

	double finalShift, finalTeta;

	Point a = Point(image.size().width / 2, 0); // first point of the axis
	Point b = Point(image.size().width / 2, image.size().height); // second point of the axis

	int middleX = image.size().width / 2;
	if ((image.size().width / 2) % 2 == 1) {
		middleX = image.size().width / 2 + 1;
	}

	int step = (int)(0.05 * image.size().width);
	for (int shift = middleX - 3 * step; shift <= middleX + 3 * step; shift += step) {
		image.copyTo(imgCopy);

		Point _a = Point(a.x = shift, a.y);
		Point _b = Point(b.x = shift, b.y);

        vector<double> grayLevel;

        int gld[511] = { 0 };
        int max = 0;
        getMeanGrayLevelDifference(image, _a, _b, gld, max, grayLevel);

        cout << "max : " << max << std::endl;

        Scalar meanDiff;
        Scalar std;
        meanStdDev(grayLevel, meanDiff, std);
        double variance = pow(std[0], 2.0);

        cout << "mean : " << meanDiff[0] << std::endl;
        cout << "std : " << std[0] << std::endl;
        cout << "var : " << variance << std::endl;

        int MEAN = getMEAN(image, _a, _b, meanDiff[0]);
        double score = MEAN / variance;

        cout << "score : " << score << std::endl;
		cout << "-------------------" << std::endl;

        imshow("histogram", histogram(gld, max)); 
		line(imgCopy, _a, _b, Scalar(255, 255, 255), 1);
		imshow("Image_crop", imgCopy);

		if (score > bestScore) {
			bestScore = score;
			shiftBestScore = shift;
		}

		waitKey(0);
	}

	cout << "Best Gray level : " << bestScore << " at " << shiftBestScore << "\n";
	//bestScoreRotate = bestScore;

	Point midPoint = Point(image.size().width / 2, image.size().height / 2);

	for (int rotateShift = -15; rotateShift <= 15; rotateShift++) {

		image.copyTo(imgCopy);

		Point _a = rotatePoint(Point(shiftBestScore, a.y), midPoint, rotateShift);
		Point _b = rotatePoint(Point(shiftBestScore, b.y), midPoint, rotateShift);
		
		vector<double> grayLevel;

		int gld[511] = { 0 };
		int max = 0;
		getMeanGrayLevelDifference(image, _a, _b, gld, max, grayLevel);

		cout << "max : " << max << std::endl;

		Scalar meanDiff;
		Scalar std;
		meanStdDev(grayLevel, meanDiff, std);
		double variance = pow(std[0], 2.0);

		cout << "mean : " << meanDiff[0] << std::endl;
		cout << "std : " << std[0] << std::endl;
		cout << "var : " << variance << std::endl;

		int MEAN = getMEAN(image, _a, _b, meanDiff[0]);
		double score = MEAN / variance;

		cout << "score : " << score << std::endl;
		cout << "-------------------" << std::endl;

		imshow("histogram", histogram(gld, max));
		line(imgCopy, _a, _b, Scalar(255, 255, 255), 1);
		imshow("Image_crop", imgCopy);

		if (score > bestScoreRotate) {
			bestScoreRotate = score;
			shiftBestScoreRotate = rotateShift;
		}
		waitKey(0);
	}

	cout << "Best Gray level rotate : " << bestScoreRotate << " at " << shiftBestScoreRotate << "\n";

	a = rotatePoint(Point(shiftBestScore, 0), midPoint, shiftBestScoreRotate);
	b = rotatePoint(Point(shiftBestScore, image.size().height), midPoint, shiftBestScoreRotate);

	detectedAxis = tLine(a, b);

	intersection(eyeLine, detectedAxis, intersect_detectedAxis_lineEye);
	intersection(referenceAxis, detectedAxis, intersect_detectedAxis_referenceAxis);

	finalShift = getLengthLine(tLine(intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye));
	finalTeta = calcTeta(intersect_detectedAxis_referenceAxis, intersect_referenceAxis_lineEye, intersect_detectedAxis_lineEye);

	cout << "FINAL SHIFT : " << finalShift << "\n";
	cout << "FINAL TETA : " << finalTeta << "\n";
	
	// convert double to string with 2 precision
	stringstream shiftStream, tetaStream;
	shiftStream << fixed << "shift " << setprecision(2) << finalShift;
	tetaStream << fixed << "teta " << setprecision(2) << finalTeta;

	// Display shift
	putText(image, shiftStream.str(), getCenterLine(tLine(intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye)), 1, 1, Scalar(0, 0, 255), 2);
	line(image, intersect_detectedAxis_lineEye, intersect_referenceAxis_lineEye, Scalar(0, 0, 255), 2);
	
	double angleDetAxis = getAngle(detectedAxis);
	putText(image, tetaStream.str(), Point(intersect_detectedAxis_referenceAxis.x, intersect_detectedAxis_referenceAxis.y - 10), 1, 1, Scalar(255, 0, 0), 2);
	ellipse(image, intersect_detectedAxis_referenceAxis, Size(25, 25), 0, angleDetAxis, angleDetAxis + finalTeta, Scalar(255, 0, 0), 2);
	/*if (finalTeta < 0) {
		ellipse(image, intersect_detectedAxis_referenceAxis, Size(25, 25), 0, getAngle(detectedAxis), getAngle(referenceAxis), Scalar(255, 0, 0), 2);
	}
	else if (finalTeta > 0) {
		ellipse(image, intersect_detectedAxis_referenceAxis, Size(25, 25), 0, getAngle(detectedAxis), getAngle(tLine(referenceAxis.p2, referenceAxis.p1)), Scalar(255, 0, 0), 2);
	}*/
	line(image, a, b, Scalar(255, 255, 255), 1);

}

int main(void) {

	string path("../images/image_crop.jpg");

	image = imread(path);

	if(image.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image = rotate(image, 10);
    
	imshow("Image crop", image);

	setMouseCallback("Image crop", onMouse);

    waitKey(0);
    return 0;
}
