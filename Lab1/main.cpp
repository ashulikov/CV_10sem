#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <bits/stdc++.h>

#define pb push_back
#define mp make_pair
#define fi first
#define se second

using namespace std;
using namespace cv;

const float INF = 1e+12;

float alpha = 5.0, beta = 10.0;
int g_mode = 1, h_mode = 1;
const int Dmax = 100;

static float gg[Dmax][Dmax][Dmax][Dmax];

float g(int d1x, int d1y, int d2x, int d2y) {
    switch (g_mode) {
        case 1: return abs(d1x - d2x) + abs(d1y + d2y);
        case 2: return sqrt((d1x - d2x) * (d1x - d2x) + (d1y - d2y) * (d1y - d2y));
        case 3: return min(beta,float(abs(d1x - d2x) + abs(d1y + d2y)));
        case 4: return min(beta,float(sqrt((d1x - d2x) * (d1x - d2x) + (d1y - d2y) * (d1y - d2y))));
    }
}

float h(Vec3b L, Vec3b R) {
    switch (h_mode) {
        case 1: return abs(L[0] - R[0]) + abs(L[1] - R[1]) + abs(L[2] - R[2]);
        case 2: return sqrt((L[0] - R[0]) * (L[0] - R[0]) + (L[1] - R[1]) * (L[1] - R[1]) + (L[2] - R[2]) * (L[2] - R[2]));
    }
}

int main(int argc, char** argv) {
    // execute program with arguments limg=C:\path\to\left\im0.png rimg=C:\path\to\right\im0.png alpha=5 beta=10 mindx=0 maxdx=5 mindy=0 maxdy=3 g=4 h=2
    static vector <pair <pair<int, int>, float> > graph[2020][Dmax][Dmax];
    int minDx, maxDx, minDy, maxDy;
    Mat Limg, Rimg;
    for (int i = 1; i < argc; ++i) {
        string tmp = argv[i];
        switch (tmp[0]) {
            case 'l':
                Limg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 'r':
                Rimg = imread(tmp.substr(5), IMREAD_COLOR);
                break;
            case 'a':
                alpha = stof(tmp.substr(6));
                break;
            case 'b':
                beta = stof(tmp.substr(5));
                break;
            case 'g':
                g_mode = stoi(tmp.substr(2));
                break;
            case 'h':
                h_mode = stoi(tmp.substr(2));
                break;
            case 'm':
                if (tmp[1]=='i'&&tmp[4]=='x') minDx = stoi(tmp.substr(6));
                if (tmp[1]=='a'&&tmp[4]=='x') maxDx = stoi(tmp.substr(6));
                if (tmp[1]=='i'&&tmp[4]=='y') minDy = stoi(tmp.substr(6));
                if (tmp[1]=='a'&&tmp[4]=='y') maxDy = stoi(tmp.substr(6));
        }
    }

    cv::Mat CloneLimg = Limg.clone();
    for (int dx = 0; dx <= maxDx - minDx; ++dx)
        for (int dy = 0; dy <= maxDy - minDy; ++dy)
            for (int ddx = 0; ddx <= maxDx - minDx; ++ddx)
                for (int ddy = 0; ddy <= maxDy - minDy; ++ddy)
                    gg[dx][dy][ddx][ddy] = alpha * g(dx, dy, ddx, ddy);

    for (int i = 0; i < Limg.rows; ++i) {
        for(int i1 = 0; i1 < 2020; i1++)
            for(int i2 = 0; i2 <= maxDx - minDx; i2++)
                for(int i3 = 0; i3 <= maxDy - minDy; i3++)
                    graph[i1][i2][i3].clear();
        for (int dx = 0; dx <= maxDx - minDx; ++dx) {
            for (int dy = 0; dy <= maxDy - minDy; ++dy) {
                graph[0][0-minDx][0-minDy].pb(mp(mp(dx,dy),0));
            }
        }
        for (int j = 0; j < Limg.cols; ++j) {
            for (int dx = 0; dx <= maxDx - minDx; ++dx) {
                for (int dy = 0; dy <= maxDy - minDy; ++dy) {
                    int jj = 2 * j + 1;
                    pair <int, int> dxy = mp(dx, dy);
                    if ((j - dx - minDx < 0) or (i - dy - minDy < 0) or (j - dx - minDx >= Limg.cols) or (i - dy - minDy >= Limg.rows)) {
                        graph[jj][dx][dy].pb(mp(dxy,INF));
                    }
                    else {
                        graph[jj][dx][dy].pb(mp(dxy,h(Limg.at<Vec3b>(i, j), Rimg.at<Vec3b>(i - dy - minDy, j - dx - minDx))));
                    }
                    if (j < Limg.cols - 1) {
                        for (int nextdx = 0; nextdx <= maxDx - minDx; ++nextdx) {
                            for (int nextdy = 0; nextdy <= maxDy - minDy; ++nextdy) {
                                graph[jj+1][dx][dy].pb(mp(mp(nextdx, nextdy),gg[dx][dy][nextdx][nextdy]));
                            }
                        }
                    }
                    else {
                        graph[jj+1][dx][dy].pb(mp(mp(0, 0),0));
                    }
                }
            }
        }
        static float d[2020][Dmax][Dmax];
        for(int i1 = 0; i1 < 2020; i1++)
            for(int i2 = 0; i2 < Dmax; i2++)
                for(int i3 = 0; i3 < Dmax; i3++)
                    d[i1][i2][i3] = INF;
        static pair<int, int> parent[2020][Dmax][Dmax];
        d[0][0-minDx][0-minDy] = 0;
        parent[0][0-minDx][0-minDy] = mp(-1,-1);
        priority_queue <pair<float,pair<int,pair<int,int> > > > q;
        q.push(mp(0, mp(0, mp(0-minDx,0-minDy))));
        pair<int,pair<int,int> > cur;
        while (!q.empty()) {
            pair<int,pair<int,int> > v = q.top().second;
            float cur_d = -q.top().first;
            q.pop();
            if (cur_d > d[v.fi][v.se.fi][v.se.se])  continue;
            if (v.fi == 2 * Limg.cols + 1) {
                cur = v;
                break;
            }
            for (size_t j=0; j<graph[v.fi][v.se.fi][v.se.se].size(); ++j) {
                pair<int, int> to = graph[v.fi][v.se.fi][v.se.se][j].first;
                float len = graph[v.fi][v.se.fi][v.se.se][j].second;
                int lvl = v.fi + 1;
                if (d[v.fi][v.se.fi][v.se.se] + len < d[lvl][to.fi][to.se]) {
                    d[lvl][to.fi][to.se] = d[v.fi][v.se.fi][v.se.se] + len;
                    parent[lvl][to.fi][to.se] = v.se;
                    q.push (mp(-d[lvl][to.fi][to.se], mp(lvl, to)));
                }
            }
        }
        while (cur.fi > 0) {
            CloneLimg.at<Vec3b>(i, cur.fi/2)[0] = parent[cur.fi][cur.se.fi][cur.se.se].fi;
            CloneLimg.at<Vec3b>(i, cur.fi/2)[1] = parent[cur.fi][cur.se.fi][cur.se.se].se;
            CloneLimg.at<Vec3b>(i, cur.fi/2)[2] = 0;
            cur.se = parent[cur.fi][cur.se.fi][cur.se.se];
            cur.fi--;
            cur.se = parent[cur.fi][cur.se.fi][cur.se.se];
            cur.fi--;
        }
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", CloneLimg );
    waitKey(0);
    imwrite("data/output.png", CloneLimg);
	return 0;
}
