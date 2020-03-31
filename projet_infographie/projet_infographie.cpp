#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h>

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

void iterateLine() {
	LineIterator it(image, noseLine.p1, noseLine.p2);
	for (int i = 0; i < it.count; i++, ++it)
	{
		Point pt = it.pos();
		cout << pt;
		cout << "\n";
	}
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

	string path("../images/image04.jpg");

	image = imread(path);

	if(image.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::imshow("Image crop", image);

	setMouseCallback("Image crop", onMouse);

	waitKey(0);
	return 0;
}
