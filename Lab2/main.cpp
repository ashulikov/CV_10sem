#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <bits/stdc++.h>
#include <Eigen\Eigen>
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
    // execute program with arguments limg=C:\path\to\left\im0.png rimg=C:\path\to\right\im1.png shift=C:\path\to\shift\map.png mindx=0 mindy=0 iter=1000000 conf=3
    vector<pair <int, int> > points;
    int minDx, minDy;
    Mat Limg, Rimg, Shift;
    int confidence = 3;
    int iterations = 1000000;
    for (int i = 1; i < argc; ++i) {
        string tmp = argv[i];
        switch (tmp[0]) {
            case 'l':
                Limg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 'r':
                Rimg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 's':
                Shift = imread(tmp.substr(6), IMREAD_COLOR);
                break;
            case 'i':
                iterations = stoi(tmp.substr(5));
                break;
            case 'c':
                confidence = stoi(tmp.substr(5));
                break;
            case 'm':
                if (tmp[1]=='i'&&tmp[4]=='x') minDx = stoi(tmp.substr(6));
                if (tmp[1]=='i'&&tmp[4]=='y') minDy = stoi(tmp.substr(6));
        }
    }
    cv::Mat CloneLimg = Limg.clone();
    cv::Mat CloneRimg = Rimg.clone();

    for (int i = confidence; i < Limg.rows - confidence; ++i) {
        for (int j = confidence; j < Limg.cols - confidence; ++j) {
            int o = 1;
            if ((Shift.at<Vec3b>(i, j)[0] == 0) and (Shift.at<Vec3b>(i, j)[1] == 0)) o = 0;
            for (int ii = -confidence; ii <= confidence; ++ii) {
                for (int jj = -confidence; jj <= confidence; ++jj) {
                    if (Shift.at<Vec3b>(i, j) != Shift.at<Vec3b>(i + ii, j + jj)) o = 0;
                }
            }
            if (o) points.pb(mp(i, j));
        }
    }

    // RANSAC Start
    Matrix3f F_ans;
    int best = 0;
    for (int it = 0; it < iterations; ++it) {
        vector<int> indices(points.size());
        iota(indices.begin(), indices.end(), 0);
        random_shuffle(indices.begin(), indices.end());
        MatrixXf m(7, 9);
        for (int i = 0; i < 7; ++i) {
            m(i, 0) = points[indices[i]].se*max(0,points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx);
            m(i, 1) = points[indices[i]].fi*max(0,points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx);
            m(i, 2) = max(0,points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx);
            m(i, 3) = points[indices[i]].se*max(0,points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy);
            m(i, 4) = points[indices[i]].fi*max(0,points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy);
            m(i, 5) = max(0,points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy);
            m(i, 6) = points[indices[i]].se;
            m(i, 7) = points[indices[i]].fi;
            m(i, 8) = 1;
        }

        JacobiSVD<MatrixXf> svd(m, ComputeFullU | ComputeFullV);
        const Eigen::MatrixXf V = svd.matrixV();
        Matrix3f F0(3, 3), F1(3, 3);
        for (int i = 0; i < 9; ++i) {
            F0(i/3, i%3) = V(7, i);
            F1(i/3, i%3) = V(8, i);
        }
        float a3 = F1.determinant();
        float a0 = F0.determinant();
        float a2;
        a2 = F1(1,0) * F1(2,1) * F0(0,2) - F1(1,0) * F1(0,1) * F0(2,2) + F1(0,0) * F1(1,1) * F0(2,2) + F1(2,0) * F0(0,1) * F1(1,2);
        a2 += F1(2,0) * F1(0,1) * F0(1,2) - F1(0,0) * F1(1,2) * F0(2,1) - F1(2,0) * F0(1,1) * F1(0,2) - F1(2,0) * F1(1,1) * F0(0,2);
        a2 +=-F1(0,0) * F0(1,2) * F1(2,1) + F0(1,0) * F1(2,1) * F1(0,2) + F1(1,0) * F0(2,1) * F1(0,2) + F0(2,0) * F1(0,1) * F1(1,2);
        a2 +=-F0(0,1) * F1(0,1) * F1(2,2) - F0(0,0) * F1(1,2) * F1(2,1) - F1(1,0) * F0(0,1) * F1(2,2) + F1(0,0) * F0(1,1) * F1(2,2);
        a2 += F0(0,0) * F1(1,1) * F1(2,2) - F0(2,0) * F1(1,1) * F1(0,2);
        float a1 = F0(0,0) * F0(1,1) * F1(2,2) + F0(0,0) * F1(1,1) * F0(2,2) + F1(2,0) * F0(0,1) * F0(1,2) - F0(1,0) * F0(0,1) * F1(1,2);
        a1 +=-F1(0,0) * F0(1,2) * F0(2,1) - F1(1,0) * F0(0,1) * F0(2,2) - F1(2,0) * F0(1,1) * F0(0,2) + F1(0,0) * F0(1,1) * F0(2,2);
        a1 += F0(1,0) * F0(2,1) * F1(0,2) + F0(1,0) * F1(2,1) * F0(0,2) + F0(2,0) * F1(0,1) * F0(1,2) - F0(1,0) * F1(0,1) * F0(2,2);
        a1 +=-F0(2,0) * F1(1,1) * F0(0,2) + F1(1,0) * F0(2,1) * F0(0,2) - F0(0,0) * F0(1,2) * F1(2,1) - F0(2,0) * F0(1,1) * F1(0,2);
        a1 += F0(2,0) * F0(0,1) * F1(1,2) - F0(0,0) * F1(1,2) * F0(2,1);
        vector <Matrix3f> F;
        if (abs(a3) < 1e-15) {
            F.pb(F1);
        }
        else {
            a0 = a0/a3;
            a1 = a1/a3;
            a2 = a2/a3;
            a3 = 1.;
            float p = (3. * a3 * a1 - a2 * a2) / 3. / a3 / a3;
            float q = (2. * a2 * a2 * a2 - 9. * a3 * a2 * a1 + 27. * a3 * a3 * a0) / 27. / a3 / a3 / a3;
            float Q = p * p * p / 27. + q * q / 4.;
            complex<float> qq = Q;
            complex<float> sqq = pow(qq,1./2.);
            complex<float> alpha = pow(-q / 2. + sqq,1./3.);
            complex<float> beta = pow(-q / 2. - sqq,1./3.);
            complex<float> y[3];
            complex<float> ii = 1.i;
            y[0] = alpha + beta;
            y[1] = (alpha + beta) / (-2.) + ii * (alpha - beta) * sqrt(3) / 2.;
            y[2] = (alpha + beta) / (-2.) - ii * (alpha - beta) * sqrt(3) / 2.;
            for (int i = 0; i<3; ++i) {
                if (abs(y[i].imag())<1e-12) {
                    float x = y[i].real() - a2 / 3. / a3;
                    Matrix3f tmp = x * F1;
                    if ((F0+tmp).determinant()!=0) {
                        Matrix3f A = F0+tmp;
                        float xx=(A(1,2)*A(0,1) - A(1,1)*A(0,2))/(A(1,0)*A(0,1)-A(1,1)*A(0,0));
                        float yy=(A(0,2)-A(0,0)*(A(1,2)*A(0,1)-A(1,1)*A(0,2))/(A(1,0)*A(0,1)-A(1,1)*A(0,0)))/A(0,1);
                        A(2,2) = xx*A(2,0) + yy*A(2,1);
                        F.pb(A);
                    }
                    else F.pb(F0+tmp);
                }
            }
        }
        for (int j = 0; j < F.size(); ++j){
            int inliner = 0;
            for (int i = 0; i < points.size(); ++i) {
                Vector3f x(points[i].se,points[i].fi,1);
                Vector3f y(max(0,points[i].se - Shift.at<Vec3b>(points[i].fi, points[i].se)[0] - minDx), max(0,points[i].fi - Shift.at<Vec3b>(points[i].fi, points[i].se)[1] - minDy), 1);
                Vector3f mult = F[j].transpose()*x;
                float d = abs(mult(0)*y(0)+mult(1)*y(1)+mult(2)*y(2)) / sqrt(mult(0) * mult(0) + mult(1) * mult(1));
                if (d<=confidence) {
                    inliner++;
                }
            }
            if (inliner > best) {
                best = inliner;
                F_ans = F[j];
            }
        }
    }
    srand(time(0));
    vector<int> indices(points.size());
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    for (int i = 0, j = 0; (j < 5)&&(i<points.size()); ++i) {
        Vector3f x(points[indices[i]].se,points[indices[i]].fi,1);
        Vector3f y(max(0,points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx), max(0,points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy), 1);
        Vector3f Lmult = F_ans*y;
        Vector3f Rmult = F_ans.transpose()*x;
        float d = abs(Rmult(0)*y(0)+Rmult(1)*y(1)+Rmult(2)*y(2)) / sqrt(Rmult(0) * Rmult(0) + Rmult(1) * Rmult(1));
        if (d<=confidence) {
            j++;
            circle(Limg,Point(x(0),x(1)), confidence, Scalar(255,0,0), -1, 8);
            circle(Rimg,Point(y(0),y(1)), confidence, Scalar(255,0,0), -1, 8);
            int lpoints[4];
            lpoints[0] = -Lmult(2)/Lmult(0);                           //Point(0, -Lmult(2)/Lmult(0));
            lpoints[1] = -Lmult(2)/Lmult(1);                           //Point(-Lmult(2)/Lmult(1), 0);
            lpoints[2] = (-Lmult(2)-Lmult(1)*(Limg.rows-1))/Lmult(0);  //Point((Limg.rows-1), (-Lmult(2)-Lmult(1)*(Limg.rows-1))/Lmult(0));
            lpoints[3] = (-Lmult(2)-Lmult(0)*(Limg.cols-1))/Lmult(1);  //Point((-Lmult(2)-Lmult(0)*(Limg.cols-1))/Lmult(1), (Limg.cols-1));
            int rpoints[4];
            vector <Point> Rp,Lp;
            rpoints[0] = -Rmult(2)/Rmult(0);                           //Point(0, -Lmult(2)/Lmult(0));
            rpoints[1] = -Rmult(2)/Rmult(1);                           //Point(-Lmult(2)/Lmult(1), 0);
            rpoints[2] = (-Rmult(2)-Rmult(1)*(Rimg.rows-1))/Rmult(0);  //Point((Limg.rows-1), (-Lmult(2)-Lmult(1)*(Limg.rows-1))/Lmult(0));
            rpoints[3] = (-Rmult(2)-Rmult(0)*(Rimg.cols-1))/Rmult(1);  //Point((-Lmult(2)-Lmult(0)*(Limg.cols-1))/Lmult(1), (Limg.cols-1));
            for (int k = 0; k < 4; ++k) {
                switch (k) {
                    case 0:
                        if ((lpoints[k] >= 0) and (lpoints[k]<=Limg.cols)) Lp.pb(Point(lpoints[k], 0));
                        if ((rpoints[k] >= 0) and (rpoints[k]<=Rimg.cols)) Rp.pb(Point(rpoints[k], 0));
                        break;
                    case 1:
                        if ((lpoints[k] >= 0) and (lpoints[k]<=Limg.rows)) Lp.pb(Point(0, lpoints[k]));
                        if ((rpoints[k] >= 0) and (rpoints[k]<=Rimg.rows)) Rp.pb(Point(0, rpoints[k]));
                        break;
                    case 2:
                        if ((lpoints[k] >= 0) and (lpoints[k]<=Limg.cols)) Lp.pb(Point(lpoints[k], Limg.rows-1));
                        if ((rpoints[k] >= 0) and (rpoints[k]<=Rimg.cols)) Rp.pb(Point(rpoints[k], Rimg.rows-1));
                        break;
                    case 3:
                        if ((lpoints[k] >= 0) and (lpoints[k]<=Limg.rows)) Lp.pb(Point(Limg.cols-1, lpoints[k]));
                        if ((rpoints[k] >= 0) and (rpoints[k]<=Rimg.rows)) Rp.pb(Point(Rimg.cols-1, rpoints[k]));
                }
            }
            line(Limg,Lp[0],Lp[1],Scalar(0,0,255),1,8);
            line(Rimg,Rp[0],Rp[1],Scalar(0,0,255),1,8);
        }
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", Limg );
    waitKey(0);
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", Rimg );
    waitKey(0);
    return 0;
}
