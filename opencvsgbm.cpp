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

using namespace cv;
using namespace std;

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

struct PathDirection
{
    int start_c,start_r;
    int end_c,end_r;
    int dc,dr;
    int dx,dy;
};

void initPaths(int height, int width,vector<PathDirection>& Paths)
{
    //8 paths from center pixel based on change in X and Y coordinates
    const int PATHS_PER_SCAN = 8;
    for(int i = 0;i < PATHS_PER_SCAN; i++)
    {
        Paths.push_back(PathDirection());
    }
    for(int i = 0 ; i < PATHS_PER_SCAN; i++)
    {
        cout << i <<endl;
        
        switch(i)
        {  
            /* 方向
                ------>        
            */
            case 0:                      
            Paths[i].start_c = 0 + 1;
            Paths[i].start_r = 0 + 1;

            Paths[i].end_c = width - 1 - 1;
            Paths[i].end_r = height - 1 - 1;

            Paths[i].dc = 1;
            Paths[i].dr = 0;

            Paths[i].dx = 1;
            Paths[i].dy = 1;
            break;
            /* 方向
                  >
                 /
                /        
            */
            case 1:
            Paths[i].start_c = 0 + 1;
            Paths[i].start_r = height - 1 - 1;

            Paths[i].end_c = width - 1;
            Paths[i].end_r = -1 + 1;

            Paths[i].dc = 1;
            Paths[i].dr = 1;

            Paths[i].dx = 1;
            Paths[i].dy = -1;
            break;

             /* 方向 
                 > 
                |
                |
            */
            case 2:
            Paths[i].start_c = 0 + 1;
            Paths[i].start_r = height - 1 - 1;

            Paths[i].end_c = width - 1;
            Paths[i].end_r = -1 + 1;

            Paths[i].dc = 0;
            Paths[i].dr = 1;

            Paths[i].dx = 1;
            Paths[i].dy = -1;
            break;

            /* 方向
               <
                \
                 \
            */
            case 3:
            Paths[i].start_c = width - 1 - 1;
            Paths[i].start_r = width - 1 - 1;

            Paths[i].end_c = -1 + 1;
            Paths[i].end_r = -1 + 1;

            Paths[i].dc = -1;
            Paths[i].dr = 1;

            Paths[i].dx = -1;
            Paths[i].dy = -1;
            break;

            /* 方向
               <------
            */
            case 4:
            Paths[i].start_c = width - 1 - 1;
            Paths[i].start_r = 0 + 1;

            Paths[i].end_c = -1 + 1;
            Paths[i].end_r = height - 1;

            Paths[i].dc = -1;
            Paths[i].dr = 0;

            Paths[i].dx = -1;
            Paths[i].dy = 1;
            break;

            /* 方向
                /
               /
              <
            */
            case 5:
            Paths[i].start_c = width - 1 - 1;
            Paths[i].start_r = 0 + 1;

            Paths[i].end_c = -1 + 1;
            Paths[i].end_r = height - 1;

            Paths[i].dc = -1;
            Paths[i].dr = -1;

            Paths[i].dx = -1;
            Paths[i].dy = 1;
            break;

             /* 方向
               | 
               |
                >
            */
            case 6:
            Paths[i].start_c = width - 1 - 1;
            Paths[i].start_r = 0 + 1;

            Paths[i].end_c = -1 + 1;
            Paths[i].end_r = height -1;

            Paths[i].dc = 0;
            Paths[i].dr = -1;

            Paths[i].dx = -1;
            Paths[i].dy = 1;
            break;

            /* 方向
               \
                \
                 >  
            */
            case 7:
            Paths[i].start_c = 0 + 1;
            Paths[i].start_r = 0 + 1;

            Paths[i].end_c = width - 1;
            Paths[i].end_r = height - 1;

            Paths[i].dc = 1;
            Paths[i].dr = -1;

            Paths[i].dx = 1;
            Paths[i].dy = 1;
            break;
        }
    }
}

// preSobelFilterCap
void preSobelFilterCap(Mat& leftImage,Mat& rightImage,uchar *tab,const int OFS)
{
    Size size = leftImage.size();

    Mat left = leftImage.clone();
    Mat right = rightImage.clone();

    uchar val0 = tab[0 + OFS]; 

    for( int y = 0; y < size.height-1; y += 2 )     //Sobel 变换
    {
        const uchar* leftsrow1 = left.ptr<uchar>(y);
        const uchar* leftsrow0 = y > 0 ? leftsrow1 - left.step : size.height > 1 ? leftsrow1 + left.step : leftsrow1;
        const uchar* leftsrow2 = y < size.height-1 ? leftsrow1 + left.step : size.height > 1 ? leftsrow1 - left.step : leftsrow1;
        const uchar* leftsrow3 = y < size.height-2 ? leftsrow1 + left.step*2 : leftsrow1;

        const uchar* rightsrow1 = right.ptr<uchar>(y);
        const uchar* rightsrow0 = y > 0 ? rightsrow1 - right.step : size.height > 1 ? rightsrow1 + right.step : rightsrow1;
        const uchar* rightsrow2 = y < size.height-1 ? rightsrow1 + right.step : size.height > 1 ? rightsrow1 - right.step : rightsrow1;
        const uchar* rightsrow3 = y < size.height-2 ? rightsrow1 + right.step*2 : rightsrow1;

        uchar* leftptr = leftImage.ptr<uchar>(y);
        uchar* leftptr2 = leftptr + leftImage.step;

        uchar* rightptr = rightImage.ptr<uchar>(y);
        uchar* rightptr2 = rightptr + rightImage.step;

        leftptr[0] = leftptr[size.width-1] = leftptr2[0] = leftptr2[size.width-1] = val0;
        rightptr[0] = rightptr[size.width-1] = rightptr2[0] = rightptr2[size.width-1] = val0;

        int d0,d1,d2,d3;
        int v0,v1;
        int x = 1;
        for( ; x < size.width-1; x++ )  //FilterCap 过程
        {
            d0 = leftsrow0[x+1] - leftsrow0[x-1];
            d1 = leftsrow1[x+1] - leftsrow1[x-1];
            d2 = leftsrow2[x+1] - leftsrow2[x-1];
            d3 = leftsrow3[x+1] - leftsrow3[x-1];
            v0 = tab[d0 + d1*2 + d2 + OFS];
            v1 = tab[d1 + d2*2 + d3 + OFS];
            leftptr[x] = (uchar)v0;
            leftptr2[x] = (uchar)v1;

            d0 = rightsrow0[x+1] - rightsrow0[x-1];
            d1 = rightsrow1[x+1] - rightsrow1[x-1];
            d2 = rightsrow2[x+1] - rightsrow2[x-1];
            d3 = rightsrow3[x+1] - rightsrow3[x-1];
            v0 = tab[d0 + d1*2 + d2 + OFS];
            v1 = tab[d1 + d2*2 + d3 + OFS];
            rightptr[x] = (uchar)v0;
            rightptr2[x] = (uchar)v1;
        }
    }
} 

void CalculateSADCost(const Mat& leftImage,const Mat& rightImage,const int SADWindows,const int disparity,vector<vector<vector<long>>>& CBTcost)
{
    int width = leftImage.cols;
    int height = leftImage.rows;

    int x1 = SADWindows;
    int y1 = SADWindows;

    int x2 = height - SADWindows;
    int y2 = width - SADWindows;

    Mat SADDisparity(height,width,0);

    int sum = 0, rightTemp = 0,leftTemp = 0;

    for(int x = x1; x < x2; x++)
    {
        for(int y = y1; y < y2; y++)
        {
            for(int d = 0;d < disparity;d++)
            {
                sum = 0;
                for(int m = x - SADWindows; m <= x + SADWindows; m++)
                {
                    for (int n = y - SADWindows; n <= y + SADWindows; n++)
                    { 
                        (m + d < width) ? leftTemp = (int)leftImage.at<uchar>(m,n+d) : leftTemp = (int)leftImage.at<uchar>(m,width - 1);
                        rightTemp = (int)rightImage.at<uchar>(m,n);
                        sum = sum + abs(leftTemp - rightTemp);
                    }
                }
                CBTcost[x][y][d] = sum/((2 *SADWindows + 1) * (2 * SADWindows + 1));
            } 
            int tempIndex = 0;
            for (int d = 1; d < disparity; d++)
            {
                if (CBTcost[x][y][d] < CBTcost[x][y][tempIndex]) 
                {  
                     tempIndex = d; 
                      
                }
            }
            SADDisparity.at<uchar>(x,y)=(tempIndex * 2);
        }          
    }
    imwrite("SADDisparity.jpg",SADDisparity);
}
    
void CensusCalculate(const Mat& leftImage,const Mat& rightImage,const int CensusWindows,vector<unsigned int>& leftCensus,vector<unsigned int>& rightCensus,const int disparity)//,unsigned int *leftCensus,unsigned int *rightCensus /* )
{

    int CensusPixelSum = (2 * CensusWindows + 1) * (2 * CensusWindows + 1);
    int bitlength = (CensusPixelSum % 32 == 0) ? (CensusPixelSum / 32) : ( CensusPixelSum / 32 + 1 );

    int width = leftImage.cols;
    int height = leftImage.rows;

    const size_t bufferSize = width * height * bitlength + disparity * bitlength;

    cout << bufferSize << endl;
    vector<unsigned int> leftCensusTemp(bufferSize);
    vector<unsigned int> rightCensusTemp(bufferSize);

    int x1 = CensusWindows;
    int y1 = CensusWindows;

    int x2 = leftImage.rows - CensusWindows;
    int y2 = leftImage.cols - CensusWindows;

    PixType CenterPixel = 0; 
    PixType neighborPixel = 0; 
    int bitCount = 0;
    int bigger = 0;

    for(int x = x1; x < x2; x++)
    {
        for(int y = y1; y < y2; y++)
        {
            CenterPixel = leftImage.at<uchar>(x, y);
            bitCount = 0;
            for(int m = x - CensusWindows; m <= x + CensusWindows; m++)
            {
                for (int n = y - CensusWindows; n <= y + CensusWindows; n++)
                {
                    bitCount++;
                    neighborPixel = leftImage.at<uchar>(m,n);

                    bigger = (neighborPixel > CenterPixel) ? 1 : 0;
                    leftCensusTemp[( x * width + y) * bitlength + bitCount/32] |= (bigger << (bitCount % 32));
                }
            }
        }
    }

    for(int x = x1; x < x2; x++)
    {
        for(int y = y1; y < y2; y++)
        {
            CenterPixel = rightImage.at<uchar>(x, y);
            bitCount = 0;
            for(int m = x - CensusWindows; m <= x + CensusWindows; m++)
            {
                for (int n = y - CensusWindows; n <= y + CensusWindows; n++)
                {
                    bitCount++;
                    neighborPixel = rightImage.at<uchar>(m,n);

                    bigger = (neighborPixel > CenterPixel) ? 1 : 0;
                    rightCensusTemp[( x * width + y) * bitlength + bitCount/32] |= (bigger << (bitCount % 32));
                }
            }
        }
    }

    leftCensus.assign(leftCensusTemp.begin(),leftCensusTemp.end()); 
    cout << leftCensus.size() << endl;
    rightCensus.assign(rightCensusTemp.begin(),rightCensusTemp.end());
}

int GetHammingWeight(unsigned int value)  
{  
    if(value == 0) return 0;  

    int a = value;  
    int b = value -1;  
    int c = 0;  

    int count = 1;  
    while(c = a & b)  
    {  
        count++;  
        a = c;  
        b = c-1;  
    }  
    return count;  
} 

void CalculateCensusCost(const vector<unsigned int> &leftCensus,const vector<unsigned int> &rightCensus,Mat& CensusDisparity,
                        const int disparity,const int CensusWindows,vector<vector<vector<long>>>& Cbuf)
{
    int CensusPixelSum = (2 * CensusWindows + 1) * (2 * CensusWindows + 1);
    int bitlength = (CensusPixelSum % 32 == 0) ? (CensusPixelSum / 32) : ( CensusPixelSum / 32 + 1 );

    int width = CensusDisparity.cols;
    int height = CensusDisparity.rows;

    const size_t bufferSize = width * height * bitlength + disparity * bitlength;

    int sum = 0;

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            for (int d = 0; d < disparity;d++)
            {
                sum = 0;
                for(int l = 0; l < bitlength; l++)
                {
                        sum += GetHammingWeight(rightCensus[(i*width+j)*bitlength + l]   
                            ^ leftCensus[(i*width+j+d)*bitlength + l]);                                         
                }
                Cbuf[i][j][d] = sum;
            }

            int  tempIndex = 0;
            for (int d = 1; d < disparity; d++)
            {
                if (Cbuf[i][j][d] < Cbuf[i][j][tempIndex])  
                 {    
                     tempIndex = d;  
                 }
            }
            CensusDisparity.at<uchar>(i,j)=(tempIndex * 2);
        }
    }
    imwrite("census.jpg",CensusDisparity);
    //imwrite("MatchLevel.jpg",MatchLevelImage); 
}

void CalculateBTHammingDistance(const vector<vector<vector<long>>>& Cbuf,vector<vector<vector<long>>>& Ccost,const int SADWindows)
{  
    int height = Cbuf.size();
    int width = Cbuf[1].size();
    int disparity = Cbuf[1][1].size();
// const size_t bufferSize = width * height * bitlength;
    int x1 = SADWindows;
    int y1 = SADWindows;

    int x2 = height - SADWindows;
    int y2 = width - SADWindows;

    Mat CensusBTDisparity(height,width,0);

    int sum = 0;

    for(int x = x1; x < x2; x++)
    {
        for(int y = y1; y < y2; y++)
        {
            for(int d = 0;d < disparity;d++)
            {
                sum = 0;
                for(int m = x - SADWindows; m <= x + SADWindows; m++)
                {
                    for (int n = y - SADWindows; n <= y + SADWindows; n++)
                    {      
                        sum = sum  + Cbuf[m][n][d];
                    }
                }
                Ccost[x][y][d] = sum/((2 *SADWindows + 1) * (2 * SADWindows + 1));
            }
           
            int tempIndex = 0;
            for (int d = 1; d < disparity; d++)
            {
                if (Ccost[x][y][d] < Ccost[x][y][tempIndex]) 
                {  
                     tempIndex = d;      
                }
            }
            CensusBTDisparity.at<uchar>(x,y)=(tempIndex * 2);
        }
    }
    imwrite("CensusBTDisparity.jpg",CensusBTDisparity);
}

void CalculateDymProgCost(const vector<vector<vector<long>>>& Cbuf,const int P1,const int P2)
{
    const int height = Cbuf.size();
    const int width = Cbuf[1].size();
    const int disparity = Cbuf[1][1].size();
    const int Pathnum = 8;

    Mat CalculateDym(height,width,0);

    vector<vector<vector<vector<int>>>> S;

    S.resize(Pathnum);
    for(int i = 0;i < Pathnum;i++)
    {
        S[i].resize(height);
        for(int j = 0;j < height;j++)
        {
            S[i][j].resize(width);
            for(int k = 0;k < width;k++)
            {
                S[i][j][k].resize(disparity+2);
                S[i][j][k][0] = S[i][j][k][disparity+1] = SHRT_MAX;
            }
        }     
    }   

    vector<int> minLr(width); //保存上一次的最小值

    vector<int> Delta(Pathnum);

    vector<int> L(Pathnum);
    
    vector<int> Lr;
    Lr.resize(disparity + 2);
   
    vector<int> Lrpre;
    Lrpre.resize(disparity + 2);
    
    vector<PathDirection>Paths;
    initPaths(height, width,Paths);

    for(int path = 0;path < Paths.size();path++)
    {
        cout << "Path" << path << endl;
        for(int i = Paths[path].start_r;i < Paths[path].end_r;i = i + Paths[path].dy)
        {
            //cout << "i" << i << "width"<< width << endl;
            for(int j = Paths[path].start_c;j < Paths[path].end_c;j = j + Paths[path].dx)
            {
                int tempIndex = 0;
                int prex = j - Paths[path].dc;
                int prey = i - Paths[path].dr;

                Lrpre.assign(S[path][prey][prex].begin(),S[path][prey][prex].end());

                int Delta = minLr[prex] + P2;
                cout << Delta << endl;

                minLr[j] = SHRT_MAX;

                for(int d = 0;d < disparity;d++)
                {
                    Lr[d+1] = Cbuf[i][j][d] + std::min((int)Lrpre[d + 1], std::min(Lrpre[d] + P1, std::min(Lrpre[d+2] + P1, Delta))) - Delta;  
                    minLr[j] = std::min(minLr[j],  Lr[d+1]);
                }
                S[path][i][j] = Lr;
                std::swap(Lr,Lrpre);
            }
        }
    }

    vector<vector<vector<long>>> Sbuf;

    Sbuf.resize(height);
    for(int i = 0;i < height; ++i)
    {
        Sbuf[i].resize(width);
        for(int j = 0;j < width;++j)
        {
            Sbuf[i][j].resize(disparity);
        }       
    }


    for (int i = 0;i < height;i++)
    {
        for(int j = 0;j < width;j++)
        {
            for(int d = 0;d < disparity;d++)
            {
                for(int path=0;path < Pathnum;path++)
                {
                    Sbuf[i][j][d] = Sbuf[i][j][d] + S[path][i][j][d + 1];
                }
            }

            int tempIndex = 0;

            for (int d = 1; d < disparity; d++)
            {
                if (Sbuf[i][j][d] < Sbuf[i][j][tempIndex]) 
                {  
                     tempIndex = d;      
                }
            }

            CalculateDym.at<uchar>(i,j)=(tempIndex * 2);       
        }
    }

    /*

    for(int i = 0;i < height;i++)
    {
        for(int j = 0;j < width;j++)
        {
            int tempIndex = 0;      
            for(int k = 0;k < 1;k++)
            {
                Delta[k] = minLr[k] + P2;
                minLr[k] = SHRT_MAX;

                for(int d = 0;d < disparity;d++)
                {
                    Lr[k][d+1] = Cbuf[i][j][d] + std::min((int)Lrpre[k][d + 1], std::min(Lrpre[k][d] + P1, std::min(Lrpre[k][d+2] + P1, Delta[k]))) - Delta[k];
                    minLr[k] = std::min(minLr[k],  Lr[k][d+1]);
                    if(Lr[k][d+1] < Lr[k][tempIndex+1])
                    {
                        tempIndex = d;
                        cout << tempIndex << endl;
                    }     
                }

                std::swap(Lr[k],Lrpre[k]);
            }          
            CalculateDym.at<uchar>(i,j)=(tempIndex * 5);
        }
    }
    */
   imwrite("CalculateDym.jpg",CalculateDym);

}

int main()
{
    Mat leftImage = imread("left.png",0);
    Mat rightImage = imread("right.png",0);

    CV_Assert( leftImage.size() == rightImage.size());

    int width = leftImage.cols;
    int height = leftImage.rows;

    const Size SADwindows = Size(9,9);

    const int OFS = 256*4, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ] = { 0 };
    const int FilterCapnum = 63;

    for( int x = 0; x < TABSZ; x++ )
    {
        tab[x] = (uchar)(x - OFS < -FilterCapnum ? 0 : x - OFS > FilterCapnum ? FilterCapnum*2 : x - OFS + FilterCapnum);
    }

    preSobelFilterCap(leftImage,rightImage,tab,OFS);

    const int disparity = 128;
    const int D = disparity;
    const int CensusWindows = 5;
    const int SADWindows2 = 5;

    vector<unsigned int>leftCensus;
    vector<unsigned int>rightCensus;

    int bitlength = 0;
    Mat CensusDisparity(height,width,0);

    CensusCalculate(leftImage,rightImage,CensusWindows,leftCensus,rightCensus,disparity);

    vector<vector<vector<long>>> Cbuf;
    vector<vector<vector<long>>> CBTcost;
    vector<vector<vector<long>>> CensusBTcost;

    Cbuf.resize(height);
    CBTcost.resize(height);
    CensusBTcost.resize(height);
    for(int i = 0;i < height; ++i)
    {
        Cbuf[i].resize(width);
        CBTcost[i].resize(width);
        CensusBTcost[i].resize(width);

        for(int j = 0;j < width;++j)
        {
            Cbuf[i][j].resize(disparity);
            CBTcost[i][j].resize(disparity);
           CensusBTcost[i][j].resize(disparity);
        }       
    }
    
    CalculateSADCost(leftImage,rightImage,SADWindows2,disparity,CBTcost);
    CalculateCensusCost(leftCensus,rightCensus,CensusDisparity,disparity,CensusWindows,Cbuf);

    CalculateBTHammingDistance(Cbuf,CensusBTcost,SADWindows2);


    const int P1 = 3;
    const int P2 = 20;

    //CalculateDymProgCost(Cbuf,P1,P2);
    CalculateDymProgCost(CensusBTcost,P1,P2);

    cout << "-----" << endl;
    imshow("left",leftImage);
    imshow("right",rightImage);
    imshow("Census",CensusDisparity);

    int key = waitKey(0);
    if(key == 27) 
        return 0;
}