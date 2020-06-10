#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <bits/stdc++.h>
#include <Eigen/Eigen>
#include <complex>

#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define float double

using namespace std;
using namespace cv;

using namespace Eigen;

const long long INF = 1000000000000000;

int main(int argc, char** argv) {
    srand(time(0));
    // execute program with arguments limg=C:\path\to\left\im0.png rimg=C:\path\to\right\im1.png x1=1 y1=1 x2=2 y2=2 x3=3 y3=3 x4=4 y4=4 u1=1 v1=1 u2=2 v2=2 u3=3 v3=3 u4=4 v4=4
    Mat Limg, Rimg;
    pair<int, int> pointL[4], pointR[4];
    for (int i = 1; i < argc; ++i) {
        string tmp = argv[i];
        switch (tmp[0]) {
            case 'l':
                Limg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 'r':
                Rimg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 'x':
                pointL[tmp[1]-'1'].fi = stoi(tmp.substr(3));
                break;
            case 'y':
                pointL[tmp[1]-'1'].se = stoi(tmp.substr(3));
                break;
            case 'u':
                pointR[tmp[1]-'1'].fi = stoi(tmp.substr(3));
                break;
            case 'v':
                pointR[tmp[1]-'1'].se = stoi(tmp.substr(3));
        }
    }
    Mat Panoram(Limg.rows, 2 * Limg.cols, CV_8UC3, Scalar(0, 0, 0));

    MatrixXf big_X(8,9);
    for (int i = 0; i < 4; ++i) {
        big_X(i * 2, 0) = -pointR[i].fi;
        big_X(i * 2, 1) = -pointR[i].se;
        big_X(i * 2, 2) = -1;
        big_X(i * 2, 3) = 0;
        big_X(i * 2, 4) = 0;
        big_X(i * 2, 5) = 0;
        big_X(i * 2, 6) = pointL[i].fi * pointR[i].fi;
        big_X(i * 2, 7) = pointL[i].fi * pointR[i].se;
        big_X(i * 2, 8) = pointL[i].fi;
        big_X(i * 2 + 1, 0) = 0;
        big_X(i * 2 + 1, 1) = 0;
        big_X(i * 2 + 1, 2) = 0;
        big_X(i * 2 + 1, 3) = -pointR[i].fi;
        big_X(i * 2 + 1, 4) = -pointR[i].se;
        big_X(i * 2 + 1, 5) = -1;
        big_X(i * 2 + 1, 6) = pointL[i].se * pointR[i].fi;
        big_X(i * 2 + 1, 7) = pointL[i].se * pointR[i].se;
        big_X(i * 2 + 1, 8) = pointL[i].se;
    }
    JacobiSVD<MatrixXf> svd(big_X, ComputeFullU | ComputeFullV);
    Matrix3f H;
    const Eigen::MatrixXf V = svd.matrixV();
    for (int i = 0; i < 9; ++i) {
        H(i/3, i%3) = V(i, 8);
    }
    cout << "Homography = \n" << H << endl;

    for (int i = 0; i < Panoram.rows; ++i) {
        for (int j = 0; j < Panoram.cols; ++j){
            Vector3f y(j, i, 1);
            Vector3f x = H.inverse() * y;
            x = x/x(2);
            if ((x(1) >= 0) and (x(1) < Rimg.rows) and (x(0) >= 0) and (x(0) < Rimg.cols)){
                if ((i >= 0) and (i < Limg.rows) and (j >= 0) and (j < Limg.cols)) {
                    Panoram.at<Vec3b>(i,j)[0] = Limg.at<Vec3b>(i ,j)[0]/2 + Rimg.at<Vec3b>(x(1),x(0))[0]/2;
                    Panoram.at<Vec3b>(i,j)[1] = Limg.at<Vec3b>(i ,j)[1]/2 + Rimg.at<Vec3b>(x(1),x(0))[1]/2;
                    Panoram.at<Vec3b>(i,j)[2] = Limg.at<Vec3b>(i ,j)[2]/2 + Rimg.at<Vec3b>(x(1),x(0))[2]/2;
                }
                else {
                    Panoram.at<Vec3b>(i,j) = Rimg.at<Vec3b>(x(1),x(0));
                }
            }
            else {
                if ((i >= 0) and (i < Limg.rows) and (j >= 0) and (j < Limg.cols)) {
                    Panoram.at<Vec3b>(i,j) = Limg.at<Vec3b>(i,j);
                }
            }
        }
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", Panoram );
    imwrite("data/panorama.png", Panoram);
    waitKey(0);
    return 0;
}
