//c++
#include <string>    
#include <vector>   
#include <iostream>
using namespace std;

//boost
#include <boost/filesystem.hpp>  

//opencv
#include <cv.h>
#include <highgui.h>
using namespace cv;

//saliency detection
#include "PreGraph.h"

//seam carving
#include "SeamCarver.h"


int image_resize(std::string image_name,std::string out_image_name,int image_size=200,double size_ratio = 1.2)
{
	std::string image_path = image_name;

	//load image	
	Mat img = imread(image_path);
	Mat img_copy;
	img.copyTo(img_copy);
	//imshow("img",img);
	//waitKey();
	
	//judege need detection and seam_carving or not
	double ratio = std::max(img.rows,img.cols)*1.0/std::min(img.rows,img.cols);
	//threshold
	if( ratio<size_ratio || img.cols <= image_size || img.rows <= image_size )
	{
		//normal resize
		cv::resize(img, img, cv::Size(image_size, image_size), (0, 0), (0, 0), cv::INTER_CUBIC);
		cv::imwrite(out_image_name,img);
		return 0;		
	}

	//saliency detection
	PreGraph SpMat;
	Mat superpixels = SpMat.GeneSp(img);
	Mat sal = SpMat.GeneSal(img);
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(tmpsuperpixels, CV_8UC3, 1.0);

	//imshow("Salicecy",tmpsuperpixels);
	//waitKey();	
	
	//pre
	for(int x=0;x<tmpsuperpixels.cols;x++)
	{
		for(int y=0;y<tmpsuperpixels.rows;y++)
		{
			//threshold
			if(tmpsuperpixels.at<uchar>(y,x)<150)
			{
				tmpsuperpixels.at<uchar>(y,x)=0;
			}
		}
	}
	//imshow("Pre",tmpsuperpixels);
	//waitKey();

	//ostu
	Mat threshold_image;  
    threshold(tmpsuperpixels, threshold_image, 0, 255,CV_THRESH_OTSU);    	
	//imshow("OSTU",threshold_image);
	//waitKey();

	//bound box detection
	vector<vector<Point>> contours;  
    vector<Vec4i> hierarchy;  
    findContours(threshold_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());  
   
	//max and mini point
	Point left_up(threshold_image.cols,threshold_image.rows),right_down(0,0);
    for(int i=0;i<contours.size();i++)  
   	{  
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
		for(int j=0;j<contours[i].size();j++)   
		{  
			//绘制出contours向量内所有的像素点  
			Point temp_point = Point(contours[i][j].x,contours[i][j].y);  
			if(temp_point.x < left_up.x)
				left_up.x = temp_point.x;
			if(temp_point.y < left_up.y)
				left_up.y = temp_point.y;
			if(temp_point.x > right_down.x)
				right_down.x = temp_point.x;
			if(temp_point.y > right_down.y)
				right_down.y = temp_point.y;
		}  
  
    }  
 	//rectangle(img,left_up,right_down,Scalar(255,255,255));
	//imshow("Object",img); 
	//waitKey(0);
	
	
	//seam_carving	
	if(img.cols>img.rows)
	{
		int width_inst = right_down.x - left_up.x;
		int width_offset = 0;
		if( width_inst < img.rows  )
		{
			width_offset = img.cols - img.rows*1.2;	
		}
		if( (img.rows<=width_inst) && (width_inst<=img.rows*1.2) )
		{
			width_offset = img.cols - width_inst;
		}
		if( img.rows*1.2<width_inst)
		{
			width_offset = img.cols - width_inst;
		}

		SeamCarver s(img_copy);
		for (int i = 0; i < width_offset; ++i)
		{

			vector<uint> seam = s.findVerticalSeam();
			s.removeVerticalSeam(seam);
		}
    		//imshow("Seam",s.getImage()); 
		//waitKey();
		Mat dst;
		cv::resize(s.getImage(), dst, cv::Size(image_size, image_size), (0, 0), (0, 0), cv::INTER_CUBIC);
		cv::imwrite(out_image_name,dst);	
	}
	else
	{
		int height_inst = right_down.y - left_up.y;
		int height_offset = 0;

		if( height_inst < img.cols  )
		{
			height_offset = img.rows - img.cols*1.2;	
		}
		if( img.cols<=height_inst && height_inst<=img.cols*1.2)
		{
			height_offset = img.rows - height_inst;
		}
		if( img.cols*1.2<height_inst)
		{
			height_offset = img.rows - height_inst;
		}

		SeamCarver s(img_copy);
		for (int i = 0; i < height_offset; ++i)
		{

			vector<uint> seam = s.findHorizontalSeam();
			s.removeHorizontalSeam(seam);
		}
		Mat dst;
		cv::resize(s.getImage(), dst, cv::Size(image_size, image_size), (0, 0), (0, 0), cv::INTER_CUBIC);
		cv::imwrite(out_image_name,dst);		
	}
	
	//cv::destroyAllWindows();
	return 1;
}

 
int get_filenames(const std::string& dir, std::vector<std::string>& filenames)  
{  
    boost::filesystem::directory_iterator start = boost::filesystem::directory_iterator(dir);
    boost::filesystem::directory_iterator di = start;

    for (; di != boost::filesystem::directory_iterator(); ++di)
    {
        filenames.push_back(di->path().filename().string()); 
    }

    return filenames.size();  
}  


int main(int argc,char* argv[])
{
	std::string in_file_path = argv[1];
	std::string out_file_path = argv[2];
	
	std::vector<std::string> file_name;
	get_filenames(in_file_path,file_name);

	for(int i=0;i<file_name.size();i++)
	{
	    cout<<"Dealing "<<file_name[i]<<endl;
	    image_resize(in_file_path+file_name[i],out_file_path+file_name[i],32);
	}
	
	return 0;
}
