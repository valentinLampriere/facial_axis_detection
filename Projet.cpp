#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("images/image_crop.jpg", IMREAD_COLOR);
	vector<Mat> imgRVB;

	if (image.empty()) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	imshow("Image 1 ", image);



	waitKey(0);
	return 0;
}