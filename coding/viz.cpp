#include <opencv2/viz.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

const double baseLength = 125.179;
const double fx = 353.454;
const double fy = 353.421;
const double u0 = 321.757;
const double v0 = 164.697;


Mat PointCloudViewer(const Mat& leftImage,const double baseLength,const double fx,const double fy,const double u0,const double v0)
{
    size_t cloudlength = leftImage.cols * leftImage.rows;
    int height = leftImage.rows;
    int width = leftImage.cols;

    Mat cloud(1,cloudlength,CV_32FC3);
    Point3f* data = cloud.ptr<cv::Point3f>();

    for(int i = 0;i < leftImage.rows;i++)
    {
         for(int j = 0;j < leftImage.cols;j++)
         {
            if((int)leftImage.at<uchar>(i,j)== 0) continue;
            data[i*height + width].z = leftImage.at<uchar>(i,j); //(double)fx * abs(baseLength)  * 0.001/ ((double)leftImage.at<uchar>(i,j));
            data[i*height + width].x = i;//(j - u0)* data[i*height + width].z / fx;
            data[i*height + width].y = j;//(i - v0)* data[i*height + width].z / fy;
         }
    }

    return cloud;
}


int main(int argn, char **argv)
{
    viz::Viz3d myWindow("Coordinate Frame");
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    Vec3f cam_pos(3.0f, 3.0f, 3.0f), cam_focal_point(3.0f, 3.0f, 2.0f), cam_y_dir(-1.0f, 0.0f, 0.0f);

    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f, -1.0f, 0.0f), Vec3f(-1.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f), cam_pos);

    Mat mleft = imread("mleft.png",0);
    Mat mright = imread("mright.png",0);
    Mat left = imread("left.png",0);

    Mat bunny_cloud = PointCloudViewer(mleft,baseLength,fx,fy,u0,v0);
    //bunny_cloud *= 5.f;
    cout << bunny_cloud.size() << endl;

    viz::WCloud cloud_widget(bunny_cloud, viz::Color::green());
    Affine3f cloud_pose = Affine3f().translate(Vec3f(0.0f, 0.0f, 3.0f));
    Affine3f cloud_pose_global = transform * cloud_pose;

    if (1)
    {
        viz::WCameraPosition cpw(0.5); // Coordinate axes
        viz::WCameraPosition cpw_frustum(Vec2f(0.889484f, 0.523599f)); // Camera frustum
        myWindow.showWidget("CPW", cpw, cam_pose);
        myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    }

    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);

    myWindow.spin();

    return 0;
}
