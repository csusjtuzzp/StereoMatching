// 双目标定
 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
   
using namespace std;  
using namespace cv;  
  
#define Camera_width 2560
#define Camera_height 720
#define Camera_FPS 30

#define Image_width 1280
#define Image_height 360
#define Image_size Size(Image_width/2,Image_height)

/* 设置图片的读取个数 */
#define frame_num 40

/* 设置角点数据 */
const int boardWidth = 10;
const int boardHeight = 7;
const int boardCorner = boardWidth * boardHeight;
const Size boardSize = Size(boardWidth,boardHeight);

 
/* 
事先标定好的左相机的内参矩阵 
fx 0 cx 
0 fy cy 
0 0  1 
*/  
const Mat cameraMatrixL = (Mat_<double>(3, 3) << 351.115, 0,       320.626,  
                                          0,       350.304, 164.036,  
                                          0,       0,       1);  
const Mat distCoeffL = (Mat_<double>(5, 1) << -0.16043, 0.0063672, 0.00012118, -0.00058769, 0.0099814);  
/* 
事先标定好的右相机的内参矩阵 
fx 0 cx 
0 fy cy 
0 0  1 
*/  
const Mat cameraMatrixR = (Mat_<double>(3, 3) << 347.642, 0,       332.368,  
                                            0,      347.593, 174.485,  
                                            0,      0,       1);  
const Mat distCoeffR = (Mat_<double>(5, 1) << -0.16890, 0.012026, 0.0011659, 0.00021586, 0.011304);  
  
  
int main(int argc, char* argv[])  
{    
    //ifstream leftname("../../readImage/readBinocularImage/Calibration_left_right_pic/Calibration_left_pic/leftname.txt");
    //ifstream rightname("../../readImage/readBinocularImage/Calibration_left_right_pic/Calibration_right_pic/rightname.txt");
    ifstream leftname("../Calibration_left_right_pic/Calibration_left_pic/leftname.txt");
    ifstream rightname("../Calibration_left_right_pic/Calibration_right_pic/rightname.txt");
    ofstream BinocularCalibrationResult("../CalibrationResult.txt",ios::out);
    FileStorage CalibrationResult("../CalibrationResult.yml",FileStorage::WRITE);
  
    string leftImageName;
    string rightImageName;

    bool isFindL,isFindR;

    vector<Point2f> leftImage_points_buf;
    vector<vector<Point2f>> leftImage_points_seq;

    vector<Point2f> rightImage_points_buf;
    vector<vector<Point2f>> rightImage_points_seq;
 
    vector<vector<Point3f>> object_points;
    vector<int> point_counts;  // 每幅图像中角点的数量


    Mat leftImageInput,rightImageInput;

    int key = waitKey(30);
    int ImageCount = 0;
    while (getline(leftname,leftImageName) && getline(rightname,rightImageName))  
    { 
	 
        leftImageInput = imread(leftImageName,CV_LOAD_IMAGE_GRAYSCALE);
        rightImageInput = imread(rightImageName,CV_LOAD_IMAGE_GRAYSCALE);
        //cout<<leftImageName<<endl;
        //cout<<rightImageName<<endl;

        isFindL = findChessboardCorners(leftImageInput, boardSize, leftImage_points_buf);  
        isFindR = findChessboardCorners(rightImageInput, boardSize, rightImage_points_buf);  
        if (isFindL == true && isFindR == true)  //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的  
        {
	    find4QuadCornerSubpix(leftImageInput, leftImage_points_buf, Size(5, 5));
	    //cornerSubPix(leftImageInput, leftImage_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));  
            //drawChessboardCorners(leftImageInput, boardSize, leftImage_points_buf, isFindL);  
            //imshow("chessboardL", leftImageInput);
  
            leftImage_points_seq.push_back(leftImage_points_buf);  
  
	    find4QuadCornerSubpix(rightImageInput, rightImage_points_buf, Size(5, 5));
            //cornerSubPix(rightImageInput, leftImage_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));  
            //drawChessboardCorners(rightImageInput, boardSize, rightImage_points_buf, isFindR);  
            //imshow("chessboardR", rightImageInput);  
            rightImage_points_seq.push_back(rightImage_points_buf);    
            ImageCount++;
	    cout<<ImageCount<<endl;
	    waitKey(100);
        }  
        else 
        {
            cout<<"Can not find all chessboards"<<endl;
        }
        
        key = waitKey(30);
        if(key != -1)
            cout<<key<<endl;
        if(key == 27 || key == 'q')
            return 0; 
    }

    int i,j,t;
    Size square_size = Size(20,20);  /* 实际测量得到的标定板上每个棋盘格的大小 */
    for (t=0;t<ImageCount;t++)
    {
        vector<Point3f> tempPointSet;
        for (i=0;i<boardSize.height;i++)
        {
           for (j=0;j<boardSize.width;j++)
            {
                Point3f realPoint;
               /* 假设标定板放在世界坐标系中z=0的平面上 */
                 realPoint.x = i*square_size.height;
                 realPoint.y = j*square_size.width;
                 realPoint.z = 0;
                 tempPointSet.push_back(realPoint);
             }
         }
        object_points.push_back(tempPointSet);
    }
     /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
    for (i=0;i<ImageCount;i++)
    {
        point_counts.push_back(boardSize.width*boardSize.height);
    }

    Mat R, T, E, F;                                         //R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵  
    vector<Mat> rvecs;                                        //旋转向量  
    vector<Mat> tvecs;                                        //平移向量  
    Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）   
    Mat mapLx, mapLy, mapRx, mapRy;                         //映射表  
    Rect validROIL, validROIR;                              //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
    
    double rms = stereoCalibrate(object_points, leftImage_points_seq, rightImage_points_seq,  
        			cameraMatrixL, distCoeffL,  
        			cameraMatrixR, distCoeffR,  
        			Size(Image_width, Image_height), R, T, E, F,  
        			CALIB_USE_INTRINSIC_GUESS,  
        			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));  

    cout << R << endl;
  
    cout << "Stereo Calibration done with RMS error = " << rms << endl; 

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, Image_size, R, T, Rl, Rr, Pl, Pr, Q,  
                  CALIB_ZERO_DISPARITY,-1,Image_size,&validROIL,&validROIR); 

    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, Image_size, CV_32FC1, mapLx, mapLy);  
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, Image_size, CV_32FC1, mapRx, mapRy); 
   
    BinocularCalibrationResult << "cameraMatrixL" << cameraMatrixL<<endl;
    BinocularCalibrationResult << "cameraMatrixR" << cameraMatrixR<<endl;

    BinocularCalibrationResult << "distCoeffL" << distCoeffL<<endl;
    BinocularCalibrationResult << "distCoeffR" << distCoeffR<<endl;

    BinocularCalibrationResult << " Rl" <<  Rl<<endl;
    BinocularCalibrationResult << "Pl" << Pl<<endl;

    BinocularCalibrationResult << " Rr" << Rl<<endl;
    BinocularCalibrationResult << "Pr" << Pr<<endl;

    CalibrationResult << "cameraMatrixL" << cameraMatrixL;
    CalibrationResult << "cameraMatrixR" << cameraMatrixR;

    CalibrationResult << "distCoeffL" << distCoeffL;
    CalibrationResult << "distCoeffR" << distCoeffR;

    CalibrationResult << "Rl" <<  Rl;
    CalibrationResult << "Pl" << Pl;

    CalibrationResult << "Rr" << Rl;
    CalibrationResult << "Pr" << Pr;
    CalibrationResult << "R" << R;
    CalibrationResult << "T" << T;

    Mat rectifyImageL, rectifyImageR; 
    remap(leftImageInput, rectifyImageL, mapLx, mapLy, INTER_LINEAR);  
    remap(rightImageInput, rectifyImageR, mapRx, mapRy, INTER_LINEAR);  

    imwrite("left.png",rectifyImageL);
    imwrite("right.png",rectifyImageR);
    Mat disp(360,640,CV_16S);
    Mat disp2(360,640,CV_8UC1); 
 //   cv::StereoSGBM sgbm;
   // cv::StereoSGBM sgbm;
   //: sgbm(rectifyImageL,rectifyImageR,disp);

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,64,11,10*11*11,40*11*11,1,1,10,100,16,StereoSGBM::MODE_SGBM);
    sgbm->compute(rectifyImageL,rectifyImageR,disp);
    //int numberOfDisparities=64; 
    disp.convertTo(disp2, CV_8U, 255/(64*16.));
    
    Mat canvas;
    canvas.create(Image_height, Image_width,CV_8UC1);
    leftImageInput.colRange(0,leftImageInput.cols).copyTo(canvas.colRange(0,leftImageInput.cols)); 
    rightImageInput.colRange(0,leftImageInput.cols).copyTo(canvas.colRange(leftImageInput.cols,canvas.cols));

    Mat canvas_rectify;
    canvas_rectify.create(Image_height, Image_width,CV_8UC1);
    rectifyImageL.colRange(0,rectifyImageL.cols).copyTo(canvas_rectify.colRange(0,rectifyImageL.cols)); 
    rectifyImageR.colRange(0,rectifyImageR.cols).copyTo(canvas_rectify.colRange(rectifyImageL.cols,canvas_rectify.cols));


    for (int i = 0; i < canvas.rows;i+=20)  
       line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8); 

    for (int i = 0; i < canvas_rectify.rows;i+=20)  
       line(canvas_rectify, Point(0, i), Point(canvas_rectify.cols, i), Scalar(0, 255, 0), 1, 8); 
    //cout<<canvas(canvas.rows,canvas.cols)
    imshow("canvas", canvas);  
    imshow("canvas_rectify", canvas_rectify); 

    imshow("disparity", disp2); 
   
    /*
    imshow("ImageL1", leftImageInput);  
    imshow("ImageR1", rightImageInput); 

    imshow("ImageL", rectifyImageL);  
    imshow("ImageR", rectifyImageR); 
    */
  
    /*
    Mat canvas;  
    double sf;  
    int w, h;  
    sf = 600. / MAX(Image_size.width, Image_size.height);  
    w = cvRound(Image_size.width * sf);  
    h = cvRound(Image_size.height * sf);  
    canvas.create(h, w * 2, CV_8UC3);  
   
    Mat canvasPart = canvas(Rect(w*0, 0, w, h));                                //得到画布的一部分  
   // resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
              cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));  
  //  rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
  
    cout << "Painted ImageL" << endl;  
    
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
  //  resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);  
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),            
              cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));  
 //   rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);  
  
    cout << "Painted ImageR" << endl; 
    */
  
    /*画上对应的线条*/  
  //  for (int i = 0; i < canvas.rows;i+=16)  
    //    line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);  
  
    //imshow("rectified", canvas);   
    /*画上对应的线条*/  
    
     
    waitKey(0);  
    system("pause");  
    return 0;  
}  

