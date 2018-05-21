#include <iostream>
#include <vector>
#include <limits>

#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

typedef uchar PixType;
typedef short CostType;
typedef short DispType;


#define Camera_width 2560
#define Camera_height 720
#define Camera_FPS 30

/*
事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/
const Mat cameraMatrixL = (Mat_<double>(3, 3) << 3.5345373698043187e+02, 0., 3.2175725352881045e+02, 0.,
                           3.5342163330009737e+02, 1.6469724862936405e+02, 0., 0., 1.);
const Mat distCoeffL = (Mat_<double>(5, 1) << -1.6536893267406616e-01, 6.1951744985478483e-03,
                        -5.2764807287869432e-04, -9.3918261299520071e-04,
                        1.4183356057020893e-02);
/*
事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/
const Mat cameraMatrixR = (Mat_<double>(3, 3) << 3.5216274739787758e+02, 0., 3.3462467168179029e+02, 0.,
                           3.5226765483357184e+02, 1.7813393213950195e+02, 0., 0., 1.);

const Mat distCoeffR = (Mat_<double>(5, 1) << -1.6353567242450973e-01, -2.7399415959978319e-03,
                        -3.2508063763371183e-04, -7.8425537645852210e-04,
                        2.1365029036439120e-02);

const Mat Rl = (Mat_<double>(3, 3) << 9.9992570984236562e-01, -2.5052808973002991e-03,
                1.1928887788335688e-02, 2.4332934830763528e-03,
                9.9997876584068357e-01, 6.0454073947402527e-03,
                -1.1943779932094362e-02, -6.0159317955562289e-03,
                9.9991057334421907e-01);

const Mat Pl = (Mat_<double>(3, 4) << 2.8968850985575989e+02, 0., 3.2263985824584961e+02, 0., 0.,
                2.8968850985575989e+02, 1.7066626739501953e+02, 0., 0., 0., 1.,
                0.);

const Mat Pr = (Mat_<double>(3, 4) << 2.8968850985575989e+02, 0., 3.2263985824584961e+02,
                -3.6263298556150701e+04, 0., 2.8968850985575989e+02,
                1.7066626739501953e+02, 0., 0., 0., 1., 0.);

const Mat Rr = (Mat_<double>(3, 3) << 9.9992570984236562e-01, -2.5052808973002991e-03,
                1.1928887788335688e-02, 2.4332934830763528e-03,
                9.9997876584068357e-01, 6.0454073947402527e-03,
                -1.1943779932094362e-02, -6.0159317955562289e-03,
                9.9991057334421907e-01);

const Mat R = (Mat_<double>(3, 3) << 9.9994697737229732e-01, 3.6103857173959105e-04,
               1.0291360218945273e-02, -4.8498642112983843e-04,
               9.9992735169944835e-01, 1.2043924256607845e-02,
               -1.0286264247903255e-02, -1.2048276826057051e-02,
               9.9987450802255429e-01);

const Mat T = (Mat_<double>(3, 1) << -1.2517963166808684e+02, 3.5631057415230738e-01,
               -2.0931087897427439e-01);

int main()
{
    //Mat leftImage = imread("left.png",0);
    //Mat rightImage = imread("right.png",0);
    VideoCapture capture(0);

    capture.set(CV_CAP_PROP_FRAME_WIDTH,Camera_width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,Camera_height);
    capture.set(CV_CAP_PROP_FPS,Camera_FPS);

    cout<<"CV_CAP_PROP_FRAME_WIDTH:"<<capture.get(CV_CAP_PROP_FRAME_WIDTH)<<endl;
    cout<<"CV_CAP_PROP_FRAME_HEIGHT:"<<capture.get(CV_CAP_PROP_FRAME_HEIGHT)<<endl;
    cout<<"CV_CAP_PROP_FPS:"<<capture.get(CV_CAP_PROP_FPS)<<endl;


    if(!capture.isOpened())
    {
        cout<<"Can not find the camera device!"<<endl;
        return -1;
    }

    Mat frame,frame_gray;

    Mat leftImage,rightImage;
    Mat templeftImage,temprightImage;
    int count = 20;
    while(count--)
    {
        capture>>frame;
        cvtColor(frame,frame_gray,CV_BGR2GRAY);
        resize(frame_gray,frame_gray,Size(),0.5,0.5);
        waitKey(30);
        //imshow("OriginPic",frame_gray);
    }



    while(1)
    {
        capture>>frame;
        cvtColor(frame,frame_gray,CV_BGR2GRAY);
        resize(frame_gray,frame_gray,Size(),0.5,0.5);
        //imshow("OriginPic",frame_gray);

        templeftImage = frame_gray(Rect(0,0,frame_gray.cols/2,frame_gray.rows));
        temprightImage = frame_gray(Rect(frame_gray.cols/2,0,frame_gray.cols/2,frame_gray.rows));

        CV_Assert( templeftImage.size() == temprightImage.size() );



        //const int width = leftImage.cols;
        //const int height = leftImage.rows;

        //cout << width << endl;
        //cout << height << endl;
        Size Image_size(templeftImage.cols,templeftImage.rows);


        Mat Q;
        Mat mapLx, mapLy, mapRx, mapRy;
        Rect validROIL, validROIR;

        stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, Image_size, R, T, Rl, Rr, Pl, Pr, Q,
                      CALIB_ZERO_DISPARITY,-1,Image_size,&validROIL,&validROIR);

        initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, Image_size, CV_32FC1, mapLx, mapLy);
        initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, Image_size, CV_32FC1, mapRx, mapRy);

        //cout << T << endl;

        Mat rectifyImageL, rectifyImageR;
        remap(templeftImage, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(temprightImage, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

        imshow("123",templeftImage);
        imshow("456",temprightImage);
        imshow("789",rectifyImageL);
        imshow("101",rectifyImageR);


        Mat canvas;
        canvas.create(templeftImage.rows, templeftImage.cols * 2,CV_8UC1);
        templeftImage.colRange(0,templeftImage.cols).copyTo(canvas.colRange(0,templeftImage.cols));
        temprightImage.colRange(0,templeftImage.cols).copyTo(canvas.colRange(templeftImage.cols,canvas.cols));


        Mat canvas_rectify;
        canvas_rectify.create(templeftImage.rows, templeftImage.cols * 2,CV_8UC1);
        rectifyImageL.colRange(0,rectifyImageL.cols).copyTo(canvas_rectify.colRange(0,rectifyImageL.cols));
        rectifyImageR.colRange(0,rectifyImageR.cols).copyTo(canvas_rectify.colRange(rectifyImageL.cols,canvas_rectify.cols));



        for (int i = 0; i < canvas.rows;i+=20)
           line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

        for (int i = 0; i < canvas_rectify.rows;i+=20)
           line(canvas_rectify, Point(0, i), Point(canvas_rectify.cols, i), Scalar(0, 255, 0), 1, 8);
        //cout<<canvas(canvas.rows,canvas.cols)



        imshow("canvas", canvas);
        imshow("canvas_rectify", canvas_rectify);

        int key = waitKey(30);
        if(key == 27) break;
        if(key == 'w')
        {
            imwrite("left.png",rectifyImageL);
            imwrite("right.png",rectifyImageR);

        }
    }


    int key = waitKey(0);
    if(key == 27)
        return 0;
}
