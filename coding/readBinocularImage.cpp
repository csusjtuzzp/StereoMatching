#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;

/* 设置摄像头参数 */

#define Camera_width 2560
#define Camera_height 720

#define Image_width 1280
#define Image_height 360
#define Camera_FPS 30

/* 设置图片的读取个数 */ 
#define frame_num 40


using namespace std;  
using namespace cv; 
 
/* 设置角点数据 */
const int boardWidth = 10;                             
const int boardHeight = 7;                            
const int boardCorner = boardWidth * boardHeight;     
const Size boardSize = Size(boardWidth, boardHeight);   


vector<Point2f> cornerL;                              //左边摄像机某一照片角点坐标集合  
vector<Point2f> cornerR;                              //右边摄像机某一照片角点坐标集合 


int main(int argc,char **argv)  
{  
    VideoCapture capture(0);
    if(!capture.isOpened())
    {
        cout<<"Can not find the camera device!"<<endl;
        return -1;
    }

    /* 设置摄像头参数 */ 
    capture.set(CV_CAP_PROP_FRAME_WIDTH,Camera_width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,Camera_height);
    capture.set(CV_CAP_PROP_FPS,Camera_FPS);
    
    cout<<"CV_CAP_PROP_FRAME_WIDTH:"<<capture.get(CV_CAP_PROP_FRAME_WIDTH)<<endl;
    cout<<"CV_CAP_PROP_FRAME_HEIGHT:"<<capture.get(CV_CAP_PROP_FRAME_HEIGHT)<<endl;
    cout<<"CV_CAP_PROP_FPS:"<<capture.get(CV_CAP_PROP_FPS)<<endl;       
    /* 定义图像 左右ROI */
    Mat frame,frame_gray;
    Mat frameROI_left,frameROI_gray_left;
    Mat frameROI_right,frameROI_gray_right;
 
    /* 保存路径文件 */
    ofstream saveleft("../../readImage/Calibration_left_pic/leftname.txt",ios::out);
    ofstream saveright("../../readImage/Calibration_right_pic/rightname.txt",ios::out);
        
	// imshow("right",frameROI_right);

	/* //测试ROI 分割是否正确---> opencv mat是从(0,0)开始计数
	cout<<(ushort)frame_gray.at<uchar>(Camera_height-1,Camera_width-1)<<endl;
	cout<<(ushort)frameROI_gray_right.at<uchar>(Camera_height-1,Camera_width/2-1)<<endl;

	cout<<(ushort)frame_gray.at<uchar>(0,Camera_width/2)<<endl;
	cout<<(ushort)frameROI_gray_right.at<uchar>(0,0)<<endl;

	cout<<(ushort)frame_gray.at<uchar>(Camera_height-1,Camera_width/2-1)<<endl;
	cout<<(ushort)frameROI_gray_left.at<uchar>(Camera_height-1,Camera_width/2-1)<<endl;

	cout<<frame_gray.rows<<endl;
	cout<<frame_gray.cols<<endl;
	cout<<frameROI_gray_right.rows<<endl;
	    cout<<frameROI_gray_right.cols<<endl;
	cout<<frameROI_gray_left.rows<<endl;
	cout<<frameROI_gray_left.cols<<endl;

	    cout<<"------------------"<<endl;
	*/

	/*
	leftimagename[4] = count_num + 1; 
	rightimagename[4] = count_num + 1; 
	*/
	int key = waitKey(30);
	int left_frame_count_num = 0;
	int right_frame_count_num = 0;

	while(key != 'q')
	{
	   capture>>frame;  
	   cvtColor(frame,frame_gray,CV_BGR2GRAY);
	   resize(frame, frame, Size(), 0.5, 0.5);
       resize(frame_gray,frame_gray,Size(),0.5,0.5);
	 //cout<<(ushort)frame.at<uchar>(0,0)<<endl;
	   imshow("OriginPic",frame);
	/* 构建左右ROI */
	   frameROI_left = frame(Rect(0,0,Image_width/2,Image_height));
	   frameROI_gray_left = frame_gray(Rect(0,0,Image_width/2,Image_height));
	// imshow("left",frameROI_left);

	   frameROI_right = frame(Rect(Image_width/2,0,Image_width/2,Image_height));
	   frameROI_gray_right = frame_gray(Rect(Image_width/2,0,Image_width/2,Image_height)); 
       std::stringstream strleft;
       std::stringstream strright;
	   if(key == 'l' && left_frame_count_num < 40)
	   {
	        strleft << "../../readImage/Calibration_left_pic/left" << left_frame_count_num + 1 << ".PNG";//图片的相对路径
	        std::cout << strleft.str() << std::endl;
	    	saveleft<<strleft.str()<<endl;
	    	imwrite(strleft.str(),frameROI_gray_left);
		    left_frame_count_num++;
		    cout<<"The left "<<left_frame_count_num<<" picture finished"<<endl;
	    }
	    if(left_frame_count_num == 40 && key == 'l') 
		    cout<<"The left Picture all finished"<<endl;
	    if(key == 'r' && right_frame_count_num < 40)
	    {
	    	 strright << "../../readImage/Calibration_right_pic/right"<< right_frame_count_num + 1 << ".PNG";//图片的相对路径
	    	 std::cout << strright.str() << std::endl;
	   	     saveright<<strright.str()<<endl;
	    	 imwrite(strright.str(),frameROI_gray_right);
		     right_frame_count_num++;
		     cout<<"The right "<<right_frame_count_num<<" picture finished"<<endl;
	    } 
	    if(right_frame_count_num == 40 && key == 'r') 
		     cout<<"The right Picture all finished"<<endl;
	    if(left_frame_count_num == 40 && right_frame_count_num == 40)
	    {
		     cout<<"Left and Right Picture all getting finished"<<endl;
		     break;
	    }              
	    key = waitKey(30);
        if(key != -1)
            cout<<key<<endl;
	    if(27 == key) return 0;   
	}

    return 0;
    
}
