#include <iostream>
#include <string>

using namespace std;


// OpenCV 库
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参
const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

//const double baseLength = 125.179;
const double baseLength = 65;
const double fx = 353.454;
const double fy = 353.421;
const double u0 = 321.757;
const double v0 = 164.697;


void transf1(const Mat& disparity, Mat& depth)
{
    const int height = disparity.rows;
    const int width = disparity.cols;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0;j < width; j++)
        {
            double tmp = (double)disparity.at<uchar>(i,j);
            cout << tmp << endl;

            if((int)tmp == 0)
                continue;
            double dis = (int)fx * abs(baseLength) / tmp;
            cout << dis << endl;
            depth.at<uchar>(i,j) = (uchar) dis;
        }
    }
    imwrite("depth.png",depth);
}

// 主函数
int main( int argc, char** argv )
{
    // 读取./data/rgb.png和./data/depth.png，并转化为点云

    // 图像矩阵
    Mat rgb;
    // 使用cv::imread()来读取图像
    // API: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#cv2.imread
    rgb = imread( "./left.png", 1 );
    //rgb = cv::imread( "./left.png", 0 );
    // rgb 图像是8UC3的彩色图像
    // depth 是16UC1的单通道图像，注意flags设置-1,表示读取原始数据不做任何修改
    Mat disparity = imread( "./ground.png", -1 );
    //depth = cv::imread( "./ground.png", -1 );
    // 点云变量
    // 使用智能指针，创建一个空点云。这种指针用完会自动释放。
    PointCloud::Ptr cloud ( new PointCloud );
    Mat depth = disparity.clone();
    transf1(disparity,depth);
    // 遍历深度图
    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            //cout << m <<endl;
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera_factor;
            p.x = (n - u0) * p.z / fx;
            p.y = (m - v0) * p.z / fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            //p.b = rgb.ptr<uchar>(m)[n];
            //p.g = rgb.ptr<uchar>(m)[n];
            //p.r = rgb.ptr<uchar>(m)[n];
            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;
    pcl::io::savePCDFile( "./pointcloud.pcd", *cloud );
    // 清除数据并退出
    cloud->points.clear();
    cout<<"Point cloud saved."<<endl;
    return 0;
}
