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

	imshow("Image crop", img);

	waitKey(0);
	return 0;
}
