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
    // execute program with arguments limg=C:\path\to\left\im0.png rimg=C:\path\to\right\im1.png shift=C:\path\to\shift\map.png mindx=0 mindy=0 iter=1000000 conf=3 diam=2
    vector<pair <int, int> > points;
    int minDx, minDy;
    Mat Limg, Rimg, Shift;
    float confidence = 3.;
    int shift_r = 3;
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
                confidence = stof(tmp.substr(5));
                break;
            case 'd':
                shift_r = stoi(tmp.substr(5));
                break;
            case 'm':
                if (tmp[1]=='i'&&tmp[4]=='x') minDx = stoi(tmp.substr(6));
                if (tmp[1]=='i'&&tmp[4]=='y') minDy = stoi(tmp.substr(6));
        }
    }
    cv::Mat CloneLimg = Limg.clone();
    cv::Mat CloneRimg = Rimg.clone();

    for (int i = shift_r; i < Limg.rows - shift_r; ++i) {
        for (int j = shift_r; j < Limg.cols - shift_r; ++j) {
            int o = 1;
            if ((Shift.at<Vec3b>(i, j)[0] == 0 - minDx) and (Shift.at<Vec3b>(i, j)[1] == 0 - minDy)) o = 0;
            for (int ii = -shift_r; ii <= shift_r; ++ii) {
                for (int jj = -shift_r; jj <= shift_r; ++jj) {
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
            m(i, 0) = points[indices[i]].se*(points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx);
            m(i, 1) = points[indices[i]].fi*(points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx);
            m(i, 2) = points[indices[i]].se - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[0] - minDx;
            m(i, 3) = points[indices[i]].se*(points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy);
            m(i, 4) = points[indices[i]].fi*(points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy);
            m(i, 5) = points[indices[i]].fi - Shift.at<Vec3b>(points[indices[i]].fi, points[indices[i]].se)[1] - minDy;
            m(i, 6) = points[indices[i]].se;
            m(i, 7) = points[indices[i]].fi;
            m(i, 8) = 1;
        }

        JacobiSVD<MatrixXf> svd(m, ComputeFullU | ComputeFullV);
        const Eigen::MatrixXf V = svd.matrixV();
        Matrix3f F0(3, 3), F1(3, 3);
        for (int i = 0; i < 9; ++i) {
            F0(i/3, i%3) = V(i, 7);
            F1(i/3, i%3) = V(i, 8);
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
        F.clear();
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
                cout << "New best F: " << best*100./points.size() << "%\n";
                F_ans = F[j];
            }
        }
        if (best*1./points.size()>0.999) break;
    }
    cout << endl << "Fundamental matrix:\n" << F_ans << endl;
    Matrix3f K;
    K << 4.2 * 72 / 25.4 / 4, 0, Limg.cols/2,
         0, 4.2 * 72 / 25.4 / 4, Limg.rows/2,
         0,                   0,           1;

    cout << "K:\n" << K << endl;
    Matrix3f E = K.transpose() * F_ans * K;
    cout << "E:\n" << E << endl;
    FullPivLU<Matrix3f> lu_decomp(E);
    cout << "rank E = " << lu_decomp.rank() << endl;
    JacobiSVD<MatrixXf> svd(E, ComputeFullU | ComputeFullV);
    const MatrixXf UU = svd.matrixU();
    const MatrixXf VV = svd.matrixV();
    Matrix3f WW;
    WW << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;
    Vector3f sigma = svd.singularValues();
    cout << "singular values E = (" << sigma(0) << ", " << sigma(1) << ", " << sigma(2) << ")" << endl;
    Matrix3f Sigm;
    Sigm << sigma(0), 0, 0,
            0, sigma(1), 0,
            0, 0, sigma(2);
    Matrix3f ETE = E.transpose() * E;
    cout << "2 * E * E^T * E - trace(E * E^T) * E = \n" << 2 * E * ETE - ETE.trace() * E << endl;
    Matrix3f cx = UU * WW * Sigm * UU.transpose();
    Matrix3f R = UU * WW.inverse() * VV.transpose();
    cout << "[c]x = \n" << cx << endl;
    cout << "R = \n" << R << endl;
    return 0;
}
