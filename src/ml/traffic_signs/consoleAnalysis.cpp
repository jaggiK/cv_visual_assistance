// console analysis
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types.h>
//#include <pcl_ conversions/pcl_ conversions.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PointStamped.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "srcsim/Console.h"

using namespace std;
using namespace cv;


// these define the box nicely if it is centered
#define MINX 370
#define MAXX 670
// for uninverted image
#define MINY 60
#define MAXY 360

// for inverted image
//#define MINY 180
//#define MAXY 480
#define ROI_WIDTH 300
#define ROI_HEIGHT 300
#define ROI_OFFSET_Y 10
#define IMAGE_MAXX 1024
#define IMAGE_MAXY 512

//#define WHITEBOX_MINX 450
//#define WHITEBOX_MAXX 590
//#define WHITEBOX_MINY 260
//#define WHITEBOX_MAXY 350
#define WHITEBOX_WIDTH 144
#define WHITEBOX_HEIGHT 88
#define WHITEBOX_BORDER 10
#define WHITEBOX_REAL_WIDTH 0.6646
#define WHITEBOX_REAL_HEIGHT 0.41477
#define WHITEBOX_METERS_PER_PIXEL 0.00464
#define WHITEBOX_REAL_CENTERX 0.
#define WHITEBOX_REAL_CENTERY 1.3696

#define LOWER_CENTER_TARGET_HEAD_FRAME_Z_METERS_PER_PIXEL 0.0037156
#define LOWER_CENTER_TARGET_HEAD_FRAME_Y_METERS_PER_PIXEL -0.003821

#define MAX_TARGET_WIDTH 120

#define MAX_EDGES 1024
#define THRESHOLD 240	// if a pixel value is higher than this, it is probably a target
#define WHITEBOX_THRESHOLD 750

#define BLUE 0
#define GREEN 1
#define RED 2

#define LEFT 0
#define RIGHT 1

#define CAMERA_BASELINE 0.07
#define FOCAL_LENGTH 610.18
#define MAX_DISPARITY 30
#define MAX_VERTICAL_MISALIGNMENT 10



/* from the urdf of the robot:
  in
  /opt/nasa/indigo/share/val_description/model/urdf/valkyrie_sim_gazebo_sync.urdf
  <link name="head">
    <inertial>
      <origin rpy="0 0 0" xyz="-0.075493 3.3383E-05 0.02774"/>
      
  <gazebo reference="head">
    <mu1>0.9</mu1>
    <mu2>0.9</mu2>
  </gazebo> 
  
    <joint name="left_camera_frame_joint" type="fixed">
    <!-- optical frame collocated with tilting DOF -->
    <origin xyz="0.0 0.035 -0.002"/>
    <parent link="head"/>
    <child link="left_camera_frame"/>
  </joint>
  <link name="left_camera_frame">
    <inertial>
      <mass value="1e-5"/>
      <!-- collocate with parent link and remove mass from it -->
      <origin xyz="-0.075493 0.035033383 0.02574"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="left_camera_optical_frame_joint" type="fixed">
    <origin rpy="-1.5708 0.0 -1.5708" xyz="0 0 0"/>
    <parent link="left_camera_frame"/>
    <child link="left_camera_optical_frame"/>
  </joint>
  <link name="left_camera_optical_frame"/>
  <gazebo reference="left_camera_frame"> 
  

  changing the origin rpy from "-1.5708 0.0 -1.5708" to "0,0,0" does not change the camera orientation or the transform, we still see:
  //camera frame XYZ: 1, 0, 0 becomes header frame XYZ: -0.000004, -0.965000, -0.002000
  
  dbarry@x1:~$ rosrun tf tf_echo left_camera_optical_frame head
At time 34.100
- Translation: [0.035, -0.002, 0.000]
- Rotation: in Quaternion [-0.500, 0.500, -0.500, -0.500]
            in RPY (radian) [-2.356, -1.571, -2.356]
            in RPY (degree) [-135.000, -90.000, -135.000]

*/

// remember that camera +x is to the right and +y is down and +z is forward, which are the directions of increasing indices in the image
// remember that head +x is forward, +y is to the left, and +z is upward
// the frame transformation from left_optical_camera_frame should show the camera's xy plane being the head's yz plane
// if that is all there was, then we would expect a left_optical_camera_frame XYZ of 1,0,0 to become 0, -1, 0 
// but we also have an offset of 3.5 cm in Y from the head frame
// with that added in, we expect a left_optical_camera_frame XYZ of 1,0,0 to become 0, -0.965, 0 
// but we also have that the camera is pitched 180 degrees and yawed 180 degrees, producing an inverted image
// with that added in, we we expect a left_optical_camera_frame XYZ of 1,0,0 to become 0, 0.965, 0 

// however, when we do the transformation, the left_optical_camera_frame XYZ of 1,0,0 becomes 0, -0.965, 0 
// which is not correct
// we have the same problem in the camera y to head z transformation, it is as if the camera is not inverted.
// what's happening???

//camera frame XYZ: 0, 0, 0 becomes header frame XYZ: 0.000000, 0.035000, -0.002000		// this is correct

//camera frame XYZ: 1, 0, 0 becomes header frame XYZ: -0.000004, -0.965000, -0.002000
//camera frame XYZ: -1, 0, 0 becomes header frame XYZ: 0.000004, 1.035000, -0.002000

//camera frame XYZ: 0, 1, 0 becomes header frame XYZ: -0.000004, 0.035000, -1.002000
//camera frame XYZ: 0, -1, 0 becomes header frame XYZ: 0.000004, 0.035000, 0.998000

//camera frame XYZ: 0, 0, 1 becomes header frame XYZ: 1.000000, 0.034996, -0.002004		// this is correct
//camera frame XYZ: 0, 0, -1 becomes header frame XYZ: -1.000000, 0.035004, -0.001996	// this is correct


class consoleAnalysis
{
private:
	ros::NodeHandle nh_;
	ros::Subscriber pointCloudSub_;
	ros::Publisher consolePub_, rvizPointPub_, headFramePointPub_;
	ros::Subscriber laserSub_;
	tf::TransformListener listener_;
	// tf::TransformListener listenerWorld_;
	tf::StampedTransform transform_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber subLeftcam_, subRightcam_;
	int numTargets_, whiteboxLeft_[2], whiteboxRight_[2], whiteboxTop_[2], whiteboxBottom_[2];
	Mat leftImage_, rightImage_;
	srcsim::Console leftConsoleMsg_, rightConsoleMsg_, previousConsoleMsg_;
	sensor_msgs::PointCloud2 rosCloud_;
	geometry_msgs::PointStamped zeroPointHeadFrame_;
	cv::Point globalWhiteboxCenter_;
	bool alreadyPrintedWhiteboxCenter_;
	//std::string pointCoudFrame_;
	CvSize sz_;


public:
bool leftImageReady_, rightImageReady_, leftImageAnalyzed_, rightImageAnalyzed_, writeImageToDisk_, distanceCalculated_;
bool newCloudReceived_, leftImageReceived_, rightImageReceived_, readyForLeftImage_, readyForRightImage_, readyForCloud_;
ros::Publisher markerPub_;
visualization_msgs::MarkerArray markerPoints_;

long scanNum_;
std_msgs::Header laserHeader_;
std::string laserHeaderFrameID_;
double laserAngle_min_, laserAngle_max_, laserAngle_increment_;
std::vector<float> laserRanges_;
	
consoleAnalysis(ros::NodeHandle &nh)
   :  nh_(nh), it_(nh)
{
	laserSub_ = nh.subscribe("/multisense/lidar_scan", 10, &consoleAnalysis::laserCallback, this);
	pointCloudSub_ = nh.subscribe<sensor_msgs::PointCloud2> ("/multisense/image_points2_color", 1, &consoleAnalysis::pointCloudCallback, this);
	subLeftcam_ = it_.subscribe("/multisense/camera/left/image_raw", 1, &consoleAnalysis::leftImageCallback, this);
	subRightcam_ = it_.subscribe("/multisense/camera/right/image_raw", 1, &consoleAnalysis::rightImageCallback, this);
	consolePub_ = nh_.advertise<srcsim::Console>("/srcsim/qual1/light", 1);
	//rvizPointPub_ = nh_.advertise<geometry_msgs::PointStamped>("rvizPoint", 10);
	//headFramePointPub_ = nh_.advertise<geometry_msgs::PointStamped>("headFramePoint", 10);
	markerPub_ = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10);
	srcsim::Console zeroMarker;
	zeroMarker.r = zeroMarker.g = zeroMarker.b = 1.0f;	
	zeroMarker.x = 2.65;
	zeroMarker.y = zeroMarker.z = 0;
	addMarker(zeroMarker);
	markerPub_.publish(markerPoints_);
	alreadyPrintedWhiteboxCenter_ = false;
	scanNum_ = 0;
 
	const std::string world_frame = "world";
	const std::string target_frame = "head";
	const std::string original_frame = "world"; //"left_camera_optical_frame";
	const ros::Time time = ros::Time(0);
	ros::Rate rate(10.0);
	while (nh.ok())
	{
		try{
			listener_.waitForTransform(target_frame, original_frame, time, ros::Duration(10.0));
			listener_.lookupTransform(target_frame, original_frame, time, transform_);
			break;
		}
		catch (tf::TransformException ex){
		  //ROS_ERROR("%s",ex.what());
		  ROS_INFO("waiting for transforms to publish");
		  ros::Duration(0.1).sleep();
		}
	}
	double headToWorldX = transform_.getOrigin().x();
	double headToWorldY = transform_.getOrigin().y();
	double headToWorldZ = transform_.getOrigin().z();
	//tf::Quaternion q = transform_.getRotation();
	double headToWorldRoll, headToWorldPitch, headToWorldYaw;
	transform_.getBasis().getRPY(headToWorldRoll, headToWorldPitch, headToWorldYaw);
	ROS_INFO("head frame from world frame: X, Y, Z = %f, %f, %f", headToWorldX, headToWorldY, headToWorldZ);
	ROS_INFO("head frame from world frame: roll, pitch, yaw = %f, %f, %f", headToWorldRoll, headToWorldPitch, headToWorldYaw);
	/*
	while (nh.ok())
	{
		try{
			listenerWorld_.waitForTransform(world_frame, target_frame, time, ros::Duration(10.0));
			listenerWorld_.lookupTransform(world_frame, target_frame, time, transformWorld_);
			break;
		}
		catch (tf::TransformException ex){
		  ROS_ERROR("%s",ex.what());
		  ROS_INFO("world or head frame is not being published");
		}
	}
	*/
	readyForLeftImage_ = false;
	readyForRightImage_ = false;
	readyForCloud_ = false;
	leftImageReady_ = false;
	rightImageReady_ = false;
	leftImageAnalyzed_ = false;
	rightImageAnalyzed_ = false;
	leftImageReceived_ = false;
	rightImageReceived_ = false;
	numTargets_ = 0;
	writeImageToDisk_ = false;
	newCloudReceived_ = false;
	distanceCalculated_ = false;
	
	
	geometry_msgs::PointStamped zeroPointWorldFrame;
	zeroPointHeadFrame_.point.x = 0.0;
	zeroPointHeadFrame_.point.y = 0.0;
	zeroPointHeadFrame_.point.z = 0.0;
	zeroPointHeadFrame_.header.frame_id = "head";
	zeroPointHeadFrame_.header.stamp = ros::Time();
	while (nh.ok())
	{
		try{
			listener_.transformPoint("world", zeroPointHeadFrame_, zeroPointWorldFrame);
			break;
		}
		catch (tf::TransformException ex){
		  //ROS_ERROR("%s",ex.what());
		  ROS_INFO("waiting for head or world frame to publish");
		  ros::Duration(0.1).sleep();
		}
	}
	
	ROS_INFO("head frame XYZ: 0, 0, 0 becomes world frame XYZ: %f, %f, %f", 
	zeroPointWorldFrame.point.x, zeroPointWorldFrame.point.y, zeroPointWorldFrame.point.z);

	geometry_msgs::PointStamped zeroPointCameraFrame;
	geometry_msgs::PointStamped cameraToHeadFrame;
	zeroPointCameraFrame.point.x = 0.0;
	zeroPointCameraFrame.point.y = 0.0;
	zeroPointCameraFrame.point.z = 0.0;
	zeroPointCameraFrame.header.frame_id = "left_camera_optical_frame";
	zeroPointCameraFrame.header.stamp = ros::Time(0);
	while (nh.ok())
	{
		try{
			listener_.transformPoint("head", zeroPointCameraFrame, cameraToHeadFrame);
			break;
		}
		catch (tf::TransformException ex){
		  //ROS_ERROR("%s",ex.what());
		  ROS_INFO("waiting for head or camera frame to publish");
		  ros::Duration(0.1).sleep();
		}
	}
	
	ROS_INFO("camera frame XYZ: 0, 0, 0 becomes header frame XYZ: %f, %f, %f", 
	cameraToHeadFrame.point.x, cameraToHeadFrame.point.y, cameraToHeadFrame.point.z);

}
void addMarker(srcsim::Console consoleMsg)
{	
	static int idNumber = 0;
	visualization_msgs::Marker marker;
	marker.ns = "marker_points";	// namespace so that this marker has a unique name
	marker.id = idNumber;
	idNumber++;
	marker.header.frame_id = "head";
	marker.header.stamp = ros::Time::now();
	marker.action = visualization_msgs::Marker::ADD;
	//marker.type = visualization_msgs::Marker::POINTS;
	marker.type = visualization_msgs::Marker::SPHERE;
	marker.lifetime = ros::Duration();
	marker.scale.x = 0.05;	// in meters
	marker.scale.y = 0.05;
	marker.scale.z = 0.05;
	// Points are white, The color of the marker is specified as a std_msgs/ColorRGBA.
	marker.color.r = consoleMsg.r;
	marker.color.g = consoleMsg.g;
	marker.color.b = consoleMsg.b;
	marker.color.a = 1.0;
	marker.pose.position.x = consoleMsg.x;
	marker.pose.position.y = consoleMsg.y;
	marker.pose.position.z = consoleMsg.z;
	marker.pose.orientation.w = 1.0;	// the other axes are set to 0 by default
	markerPoints_.markers.push_back(marker);		
}
	
void leftImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	
	//geometry_msgs::PointStamped cameraFramePointStamped;
	//cameraFramePointStamped.point.x = 1.00;
	//cameraFramePointStamped.point.y = -1.0;
	//cameraFramePointStamped.point.z = 2.0;
	//cameraFramePointStamped.header.frame_id = "left_camera_optical_frame";
	//rvizPointPub_.publish(cameraFramePointStamped);
	
	
	if (!readyForLeftImage_) return;
	readyForLeftImage_ = false;
	leftImageReceived_ = true;
	leftImageAnalyzed_ = false;
	//ROS_INFO("left image received, analyzing now");
	try
	{
		leftImage_= cv_bridge::toCvCopy(msg, "bgr8")->image;
	}
		catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		leftImageReceived_ = false;
		readyForLeftImage_ = true;	// allow another image to process
		return;
	}	
	//cv::flip(leftImage_, leftImage_, -1);	// image comes through inverted
	//analyzeImage(leftImage_, RIGHT);
}

void rightImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	if (!readyForRightImage_) return; // only deal with one image at a time
	//ROS_INFO("right image received, analyzing now");
	readyForRightImage_ = false;
	rightImageReceived_ = true;
	rightImageAnalyzed_ = false;
	try
	{
		rightImage_= cv_bridge::toCvCopy(msg, "bgr8")->image;
	}
		catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		rightImageReceived_ = false;	// allow another image to process
		return;
	}	
	cv::flip(rightImage_, rightImage_, -1);	// image comes through inverted	
	//analyzeImage(rightImage_, RIGHT);
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	if (!readyForCloud_) return;
	readyForCloud_ = false;
	newCloudReceived_ = true;
	rosCloud_ = *msg;
	//pointCoudFrame_ = msg->header.frame_id;
	//ROS_INFO("new point cloud received, header on point cloud says %s", msg->header.frame_id.c_str());	
	//calculateDistance();
}
		
bool analyzeImage(int side)
{
	srcsim::Console analyzeImageConsoleMsg;	
	Mat showImage;
	cv::circle(showImage,cv::Point(IMAGE_MAXX / 2, IMAGE_MAXY / 2),5,cv::Scalar(0,255,0),5);	// put a dot at the image center
	if (side == LEFT)
	{
		//ROS_INFO("analyzing left image which has %d columns (X) and %d rows (Y)", leftImage_.cols, leftImage_.rows);
		showImage = leftImage_.clone();
	}		
	else 
	{
		//ROS_INFO("analyzing right image which has %d columns (X) and %d rows (Y)", rightImage_.cols, rightImage_.rows);
		showImage = rightImage_.clone();
	}

	/*
	cv::Point lt;
	lt.x = MINX;
	lt.y = MINY;
	cv::Point br;
	br.x = MAXX;
	br.y = MAXY;
	cv::rectangle(showImage, lt, br, cv::Scalar(0,0,255));
	if (side == LEFT) cv::imshow("left_result", showImage);
	else cv::imshow("right_result", showImage);
	cv::waitKey(0);
	*/
	
	// first find the big white box
	cv::Point whiteboxCenter;							
	if (!findWhitebox(showImage, side, &whiteboxCenter.x, &whiteboxCenter.y))
	{
		ROS_WARN("unable to find the white box for this image");
		leftImageReady_ = false;
		rightImageReady_ = false;
		rightImageAnalyzed_ = true;
		leftImageAnalyzed_ = true;
		return false;
	}	
	globalWhiteboxCenter_ = whiteboxCenter;
	int startX, finishX, startY, finishY;
	startX = whiteboxCenter.x - (ROI_WIDTH / 2);
	
	//startY = (whiteboxCenter.y - ((ROI_HEIGHT / 2)) + 30); // inverted
	startY = (whiteboxCenter.y - ((ROI_HEIGHT / 2)) - 30);
	
	finishX = whiteboxCenter.x + (ROI_WIDTH / 2);
	finishY = (whiteboxCenter.y + ((ROI_HEIGHT / 2)));	
	
	cv::rectangle(showImage, cv::Point(startX, startY), cv::Point(finishX, finishY), cv::Scalar(0,0,255));	

	// now analyze image			
	vector<Mat> planes_BGR;			// we expect to recieve color images, CV_8UC3
	if (side == LEFT) split(leftImage_, planes_BGR); // split into separate color planes (B G R)
	else split(rightImage_, planes_BGR);	
	// remember that openCV defaults to BGR color ordering rather than RGB
	Mat tmpImage;
	tmpImage.create(planes_BGR[0].size(), planes_BGR[0].type()); // single color
		
	int value = 0, centerColor = 0, centerX, centerY, targetWidth;
	int centerValueBlue, centerValueGreen, centerValueRed;
	double targetSize;
	bool targetFound = false;
	int threshold = THRESHOLD;		
	
	for (int k = 0; k < 3; k++)
	{
		bool topLeftFound = false, topRightFound = false, bottomFound = false;
		int leftCol, rightCol, topRow, bottomRow;;
		
		tmpImage = planes_BGR[k];	// applies the header

		CV_Assert(tmpImage.depth() == CV_8U);  // accept only uchar images 
		if (tmpImage.rows < MAXY || tmpImage.cols < MAXX)
		{
			ROS_WARN("image size is too small for analysis, number of rows, cols = %d, %d", tmpImage.rows, tmpImage.cols);
			leftImageReady_ = false;
			rightImageReady_ = false;
			rightImageAnalyzed_ = true;
			leftImageAnalyzed_ = true;
			return false;
		}

		for(int j = startY; j < finishY; j++)
		{
			const uchar* currentRow = tmpImage.ptr<uchar>(j);
			for(int i = startX; i < finishX; i++)
			{
				// exclude whitebox area
				if (j >= whiteboxTop_[side] && j <= whiteboxBottom_[side] && i >= whiteboxLeft_[side]  && i <= whiteboxRight_[side])
					 i = whiteboxRight_[side] + 1;
					 
				value = currentRow[i];
				if (topLeftFound && topRightFound && value < threshold)
				{
					bottomRow = j;
					centerY = topRow + ((bottomRow - topRow) / 2.);
					
					// for slanted targets we need to find the horizontal center again
					tmpImage = planes_BGR[k];
					const uchar* centerRow = tmpImage.ptr<uchar>(centerY);
					for(int m = 2; m < MAX_TARGET_WIDTH; m++)
					{
						if (leftCol + m < MAXX)
						{
							value = centerRow[leftCol + m];
							if (value < threshold)
							{
								targetWidth = m;
								centerX = leftCol + (targetWidth / 2);
								m = MAX_TARGET_WIDTH;
							}
						}
					}					
					targetSize = targetWidth * (bottomRow - topRow);
					centerColor = k;					
					targetFound = true;
					ROS_INFO("target found with color = %d, value = %d bottom center is at x, y = %d, %d", k, value, centerX, j); 
				}
				if (value > threshold && (!targetFound))
				{
					if (!topLeftFound)
					{					
						ROS_INFO("top left corner found with color = %d, value = %d at x, y = %d, %d", k, value, i, j); 
						leftCol = i;
						topRow = j;
						/*
						if (leftCol > 620 && leftCol < 650 && topRow > 280 && topRow < 300)
						{
							ROS_INFO("Big lower right target found");
							centerX = whiteboxCenter.x + 100;
							centerY = whiteboxCenter.y + 82;
							targetSize = 2870;
							centerColor = k;					
							targetFound = true;
							//ROS_INFO("target found with color = %d", k);
							//cv::imshow("left_result", showImage);
							//cv::waitKey(0);
						}
						
						else if (leftCol > 470 && leftCol < 490 && topRow > 290 && topRow < 309)
						{
							ROS_INFO("Big lower center target found");
							centerX = whiteboxCenter.x;
							centerY = whiteboxCenter.y + 82;
							targetSize = 2870;
							centerColor = k;					
							targetFound = true;
							//ROS_INFO("target found with color = %d", k);
							//cv::imshow("left_result", showImage);
							//cv::waitKey(0);
						}
						else
						*/   
						{
							startX = i;
							cv::Point topLeftPoint;
							topLeftPoint.x = leftCol;
							topLeftPoint.y = topRow;
						
							i += 1; // so that we don't stop looking too close to the top left corner
							j += 2; // this will make the next point we look at be two cols over and 2 cols down
							topLeftFound = true;
						}							
					}		
				}
				else if (!targetFound)
				{
					if (topLeftFound)
					{
						if (j == topRow + 2 && (!topRightFound))
						{
							rightCol = i;
							targetWidth = i - leftCol;
							centerX = (leftCol + (targetWidth / 2.));
							ROS_INFO("top right corner found with color = %d, value = %d at x, y = %d, %d", k, value, i, topRow);
							startX = startX + ((rightCol - startX) / 2.);	// next scan the horizontal center pixel in each row until it is below threshold
							finishX = startX + 1; // just need to look at a single pixel
							i = finishX;	// go to the next row
							topRightFound = true;
							
						} 
					}
				}
				if (targetFound)
				{
					i = finishX;
					j = finishY;
					k = 3;
				}
			}
		}
	}
	if (targetFound)
	{
		// get values for target center point					
		tmpImage = planes_BGR[BLUE];
		const uchar* centerRowBlue = tmpImage.ptr<uchar>(centerY);
		centerValueBlue = centerRowBlue[centerX];
		tmpImage = planes_BGR[GREEN];
		const uchar* centerRowGreen = tmpImage.ptr<uchar>(centerY);
		centerValueGreen = centerRowGreen[centerX];
		tmpImage = planes_BGR[RED];
		const uchar* centerRowRed = tmpImage.ptr<uchar>(centerY);
		centerValueRed = centerRowRed[centerX];

		cv::Point2f centerPoint;
		centerPoint.x = centerX;
		centerPoint.y = centerY;
		analyzeImageConsoleMsg.x = centerX;
		analyzeImageConsoleMsg.y = centerY;
		analyzeImageConsoleMsg.z = 1;
		analyzeImageConsoleMsg.b = ((double) centerValueBlue) / 255.;
		analyzeImageConsoleMsg.g = ((double) centerValueGreen) / 255.;
		analyzeImageConsoleMsg.r = ((double) centerValueRed) / 255.;		
		//cv::circle(showImage,centerPoint,3,cv::Scalar(centerValueBlue,centerValueGreen,centerValueRed),3);
		cv::circle(showImage,centerPoint,3,cv::Scalar(0, 0, 0),3);
		if (side == LEFT)
		{
			//ROS_INFO("left target found, color is %d, pixel center is at x,y: %d, %d, size is %f", centerColor, centerX, centerY, targetSize);
			leftConsoleMsg_ = analyzeImageConsoleMsg;
		}
		else 
		{
			//ROS_INFO("right target found, color is %d, pixel center is at x,y: %d, %d, size is %f", centerColor, centerX, centerY, targetSize);
			rightConsoleMsg_ = analyzeImageConsoleMsg;
		}
		//ROS_INFO("Color values are RBG: %f, %f, %f", analyzeImageConsoleMsg.r, analyzeImageConsoleMsg.g, analyzeImageConsoleMsg.b);
	}
	else
	{
		//if (side == LEFT) ROS_INFO("no target found in left image");
		//else ROS_INFO("no target found in right image");
		leftImageReady_ = false;
		rightImageReady_ = false;
		rightImageAnalyzed_ = true;
		leftImageAnalyzed_ = true;
		return false;
	}
	if (leftImageReady_)
	{			
		if (fabs(leftConsoleMsg_.b - rightConsoleMsg_.b) > 0.1
			|| fabs(leftConsoleMsg_.g - rightConsoleMsg_.g) > 0.1
			|| fabs(leftConsoleMsg_.r - rightConsoleMsg_.r) > 0.1)
		{
			ROS_INFO("bgr mismatch between stereo targets");
			leftImageReady_ = false;
			rightImageReady_ = false;
			leftImageAnalyzed_ = true;
			rightImageAnalyzed_ = true;
			return false;
		}
		if (fabs(leftConsoleMsg_.x - rightConsoleMsg_.x) > MAX_DISPARITY
			|| fabs(leftConsoleMsg_.y - rightConsoleMsg_.y) > MAX_VERTICAL_MISALIGNMENT)
		{
			ROS_INFO("location mismatch between stereo targets");
			leftImageReady_ = false;
			rightImageReady_ = false;
			leftImageAnalyzed_ = true;
			rightImageAnalyzed_ = true;
			return false;
		}
		cv::imshow("right_result", showImage);
		ROS_INFO("right image ready, center is at x,y: %d, %d", centerX, centerY);
		rightImageReady_ = true;
	}
	else
	{
		cv::imshow("left_result", showImage);
		ROS_INFO("left image ready, center is at x,y: %d, %d", centerX, centerY);
		leftImageReady_ = true;	
	}
	return true;
}

bool findWhitebox(Mat image, int side, int *centerX, int *centerY)
{
	vector<Mat> planes_BGR;			// we expect to recieve color images, CV_8UC3
	split(image, planes_BGR); // split into separate color planes (B G R)
	bool result = false;
	for(int j = MINY; j < MAXY; j++)
	{
		const uchar* currentRowBlue = planes_BGR[0].ptr<uchar>(j);
		const uchar* currentRowGreen = planes_BGR[1].ptr<uchar>(j);
		const uchar* currentRowRed = planes_BGR[2].ptr<uchar>(j);
		for(int i = MINX; i < MAXX; i++)
		{
			int value = currentRowBlue[i] + currentRowGreen[i] + currentRowRed[i];
			if (value > WHITEBOX_THRESHOLD)
			{
				const uchar* whiteboxRowBlue = planes_BGR[0].ptr<uchar>(j + 5);
				const uchar* whiteboxRowGreen = planes_BGR[1].ptr<uchar>(j + 5);
				const uchar* whiteboxRowRed = planes_BGR[2].ptr<uchar>(j + 5);
				int indexMidbox = i + (WHITEBOX_WIDTH / 2);
				int indexPastbox = i + (WHITEBOX_WIDTH + 5);
				if ((whiteboxRowBlue[indexMidbox] + whiteboxRowGreen[indexMidbox] + whiteboxRowRed[indexMidbox] > WHITEBOX_THRESHOLD)
					&& (whiteboxRowBlue[indexPastbox] + whiteboxRowGreen[indexPastbox] + whiteboxRowRed[indexPastbox] < WHITEBOX_THRESHOLD))
				
				{
					//ROS_INFO("found the white box top left with value = %d, at (x, y) = %d, %d", value, i, j);
					result = true;
					whiteboxLeft_[side] = i;
					whiteboxTop_[side] = j;
					for (int k = i; k < i + WHITEBOX_WIDTH + WHITEBOX_BORDER; k++)
					{
						value = whiteboxRowBlue[k] + whiteboxRowGreen[k] + whiteboxRowRed[k];
						if (value < WHITEBOX_THRESHOLD)
						{
							//ROS_INFO("found right edge of whitebox at x = %d", k-1);
							//ROS_INFO("whitebox width = %d", (k - 1) - whiteboxLeft_[side]);
							k = i + WHITEBOX_WIDTH + WHITEBOX_BORDER;
						}
					}
					i = MAXX;
					j = MAXY;	
				}
			}
		}
	}
	if (!result) return false;
	
	// display the box
	cv::Point whiteboxCenter;
	whiteboxCenter.x = whiteboxLeft_[side] + (WHITEBOX_WIDTH / 2);
	whiteboxCenter.y = whiteboxTop_[side] + (WHITEBOX_HEIGHT / 2);
	whiteboxRight_[side] = whiteboxLeft_[side] + WHITEBOX_WIDTH + WHITEBOX_BORDER;
	whiteboxBottom_[side] = whiteboxTop_[side] + WHITEBOX_HEIGHT + WHITEBOX_BORDER;
	whiteboxLeft_[side] = whiteboxLeft_[side] - WHITEBOX_BORDER;
	whiteboxTop_[side] = whiteboxTop_[side] - WHITEBOX_BORDER;
	
	if (!alreadyPrintedWhiteboxCenter_)
	{
		ROS_INFO("whitebox center: %d, %d", whiteboxCenter.x, whiteboxCenter.y);
		alreadyPrintedWhiteboxCenter_ = true;
	}
	cv::rectangle(image, cv::Point(whiteboxLeft_[side], whiteboxTop_[side]), cv::Point(whiteboxRight_[side], whiteboxBottom_[side]), cv::Scalar(255,0,255));
	cv::circle(image,whiteboxCenter,3,cv::Scalar(255,0,0),3);
	*centerX = whiteboxCenter.x;
	*centerY = whiteboxCenter.y;
	return true;	
}

bool calculateDistance()
{
	bool result = true;
	double disparity = fabs(leftConsoleMsg_.x - rightConsoleMsg_.x);
	int correctionCameraFrameX = 0, correctionCameraFrameY = 0;
	if (disparity > 0) 
	{
		srcsim::Console averageMsg, calibratedMsg;
		averageMsg.b = (rightConsoleMsg_.b + leftConsoleMsg_.b) / 2.;
		if (averageMsg.b > .98) averageMsg.b = 1.;
		else if (averageMsg.b < .02) averageMsg.b = 0;
		averageMsg.g = (rightConsoleMsg_.g + leftConsoleMsg_.g) / 2.;
		if (averageMsg.g > .98) averageMsg.g = 1.;
		else if (averageMsg.g < .02) averageMsg.g = 0;
		averageMsg.r = (rightConsoleMsg_.r + leftConsoleMsg_.r) / 2.;
		if (averageMsg.r > .98) averageMsg.r = 1.;
		else if (averageMsg.r < .02) averageMsg.r = 0;
		
		averageMsg.x = (rightConsoleMsg_.x + leftConsoleMsg_.x) / 2.;
		averageMsg.y = (rightConsoleMsg_.y + leftConsoleMsg_.y) / 2.;
		averageMsg.z = (CAMERA_BASELINE * FOCAL_LENGTH) / disparity;
		
				
		if (fabs(previousConsoleMsg_.x - averageMsg.x) < 3
			&& fabs(previousConsoleMsg_.y - averageMsg.y) < 3
			&& fabs(previousConsoleMsg_.r - averageMsg.r) < 0.05
			&& fabs(previousConsoleMsg_.g - averageMsg.g) < 0.05
			&& fabs(previousConsoleMsg_.b - averageMsg.b) < 0.05)
		{
			ROS_INFO("same point seen again");
			return false;
		}
		
		calibratedMsg.b = averageMsg.b;
		calibratedMsg.g = averageMsg.g;
		calibratedMsg.r = averageMsg.r;
		calibratedMsg.x = (averageMsg.x - (IMAGE_MAXX / 2)) / FOCAL_LENGTH;
		calibratedMsg.y = (averageMsg.y - (IMAGE_MAXY / 2)) / FOCAL_LENGTH;
		calibratedMsg.z = averageMsg.z;
		ROS_INFO("using individual images in averaged camera optical frame, disparity = %f, distance = %f, x,y = %f, %f",
		 disparity, averageMsg.z, calibratedMsg.x, calibratedMsg.y);
		 
		srcsim::Console finalConsoleMsg;
		finalConsoleMsg.b = averageMsg.b;
		finalConsoleMsg.g = averageMsg.g;
		finalConsoleMsg.r = averageMsg.r;
		bool pValueFound = false;

		pcl::PointCloud<pcl::PointXYZRGB> pclCloud;
		pcl::fromROSMsg<pcl::PointXYZRGB>(rosCloud_, pclCloud);
		//ROS_INFO("Cloud size: %d, %d", pclCloud.width, pclCloud.height);
		//ROS_INFO("organized: %d", pclCloud.isOrganized());
		//ROS_INFO("dense: %d", pclCloud.is_dense);
		int coordX = (int) leftConsoleMsg_.x;	// want to use left message, not average message, so that we can transform to the head frame
		int coordY = (int) leftConsoleMsg_.y;
		//ROS_INFO("in pixels, avgX, avgY = %d, %d", coordX, coordY);
		pcl::PointXYZRGB p = pclCloud(coordX, coordY);
		int indexX, indexY;
		if (pcl::isFinite(p))
		{
			finalConsoleMsg.x = p.x;
			finalConsoleMsg.y = p.y;
			finalConsoleMsg.z = p.z;
			ROS_INFO("left camera optical frame at pixels x, y: %d, %d", coordX, coordY);
			//ROS_INFO_STREAM("found correspnding XYZ with finite p: " << p);
			pValueFound = true;
		}
		else
		{
			finalConsoleMsg.x = 0.;
			finalConsoleMsg.y = 0.;
			finalConsoleMsg.z = 0.;
			/*
			int deltaX = -1, deltaY = -1;
			for (int i = 0; i < 100; i++)
			{
				for (int signValX = -1; signValX < 2; signValX = signValX + 2)
				{
					indexX = coordX + (signValX * i); 
					for (int j = 0; j < i + 2; j++)
					{
						for (int signValY = -1; signValY < 2; signValY = signValY + 2)
						{
							indexY = coordY + (signValY * j);
							if (indexX >= 0 && indexX <= MAXX && indexY >= 0 && indexY <= MAXY)
							{				
								pcl::PointXYZRGB p = pclCloud(indexX, indexY);
								if (pcl::isFinite(p))
								{
									correctionCameraFrameX = indexX - coordX;
									correctionCameraFrameY = indexY - coordY;
									ROS_INFO("left camera optical frame at pixels at x, y: %d, %d", indexX, indexY);
									ROS_INFO("which are offset from actual center by x, y: %d, %d", correctionCameraFrameX, correctionCameraFrameY);
									//ROS_INFO_STREAM("found correspnding XYZ with finite p: " << p);
									finalConsoleMsg.x = p.x;
									finalConsoleMsg.y = p.y;
									finalConsoleMsg.z = p.z;
									pValueFound = true;
									signValY = 2;
									signValX = 2;
									i = 100;
									j = 102;
								}
								//else
								//{
								//	ROS_INFO("p value does not exist at i,j: %d, %d", i, j);
								//}
							}
						}
					}
				}
			}
			*/
		}
		
		
		
		if (pValueFound)
		{
			// get distance from laser
			moveHead(finalConsoleMsg.y);	// get the head lined up with the light to get an accurate distance
			// wait for this to return
			double laserRange = getLaserRange(finalConsoleMsg.x);
			// sanity check
			//if (laserRange > 2.0 && laserRange < 3.5) finalConsoleMsg.z = laserRange;
			
			// check in rviz that the point is where we think it is
			geometry_msgs::PointStamped cameraFramePointStamped;
			cameraFramePointStamped.point.x = finalConsoleMsg.x;
			cameraFramePointStamped.point.y = finalConsoleMsg.y;
			cameraFramePointStamped.point.z = finalConsoleMsg.z;				
			cameraFramePointStamped.header.frame_id = "left_camera_optical_frame";
			//rvizPointPub_.publish(cameraFramePointStamped);
			ROS_INFO("left camera optical frame XYZ = %f, %f, %f", finalConsoleMsg.x, finalConsoleMsg.y, finalConsoleMsg.z);
			
			geometry_msgs::PointStamped headFramePointStamped;
			headFramePointStamped.header.frame_id = "head";
			while (nh_.ok())
			{
				try{
					listener_.transformPoint("head", cameraFramePointStamped, headFramePointStamped);
					break;
				}
				catch (tf::TransformException ex){
				  //ROS_ERROR("%s",ex.what());
				  ROS_INFO("waiting for head or camera frame to publish");
				  ros::Duration(0.1).sleep();
				}
			}
			
			//headFramePointPub_.publish(headFramePointStamped);
			finalConsoleMsg.x = headFramePointStamped.point.x;
			finalConsoleMsg.y = headFramePointStamped.point.y; // + (LOWER_CENTER_TARGET_HEAD_FRAME_Y_METERS_PER_PIXEL * ((double) correctionCameraFrameX));
			finalConsoleMsg.z = headFramePointStamped.point.z; // + (LOWER_CENTER_TARGET_HEAD_FRAME_Z_METERS_PER_PIXEL * ((double) correctionCameraFrameY));

			/*
			const std::string world_frame1 = "world";
			const std::string target_frame1 = "head";
			const std::string original_frame1 = "left_camera_optical_frame";	
			const ros::Time time1 = ros::Time(0);		
			while (nh_.ok())
			{
				try{
					listener_.waitForTransform(world_frame1, target_frame1, time1, ros::Duration(10.0));
					listener_.lookupTransform(world_frame1, target_frame1, time1, transformWorld_);
					listener_.transformPoint("world", headFramePointStamped, worldFramePointStamped);
					break;
				}
				catch (tf::TransformException ex){
				  ROS_ERROR("%s",ex.what());
				  ROS_INFO("world frame is not being published fast enough");
				}
			}
			*/
			geometry_msgs::PointStamped worldFramePointStamped, zeroPointWorldFrame;
			headFramePointStamped.header.stamp = ros::Time(0);
			zeroPointWorldFrame.header.stamp = ros::Time(0);
			while (nh_.ok())
			{
				try{
					listener_.transformPoint("world", headFramePointStamped, worldFramePointStamped);
					break;
				}
				catch (tf::TransformException ex){
				  //ROS_ERROR("%s",ex.what());
				  ROS_INFO("waiting for head or world frame to publish");
				  ros::Duration(0.1).sleep();
				}
			}
			
			zeroPointHeadFrame_.point.x = 0.0;
			zeroPointHeadFrame_.point.y = 0.0;
			zeroPointHeadFrame_.point.z = 0.0;
			zeroPointHeadFrame_.header.frame_id = "head";
			zeroPointHeadFrame_.header.stamp = ros::Time();
			while (nh_.ok())
			{
				try{
					listener_.transformPoint("world", zeroPointHeadFrame_, zeroPointWorldFrame);
					break;
				}
				catch (tf::TransformException ex){
				  //ROS_ERROR("%s",ex.what());
				  ROS_INFO("waiting for head or world frame to publish again");
				  ros::Duration(0.1).sleep();
				}
			}		
							
			srcsim::Console worldConsoleMsg;
			worldConsoleMsg.x = worldFramePointStamped.point.x;
			worldConsoleMsg.y = worldFramePointStamped.point.y;
			worldConsoleMsg.y = worldFramePointStamped.point.z;
			srcsim::Console worldZeroPointConsoleMsg;
			worldZeroPointConsoleMsg.x = zeroPointWorldFrame.point.x;
			worldZeroPointConsoleMsg.y = zeroPointWorldFrame.point.y;
			worldZeroPointConsoleMsg.y = zeroPointWorldFrame.point.z;
			
			
			
			//ROS_INFO("console message published");
			consolePub_.publish(finalConsoleMsg); 
			addMarker(finalConsoleMsg);
			ROS_INFO("head frame XYZ = %f, %f, %f", finalConsoleMsg.x, finalConsoleMsg.y, finalConsoleMsg.z);
			ROS_INFO("world frame from head XYZ = %f, %f, %f", worldConsoleMsg.x, worldConsoleMsg.y, worldConsoleMsg.z);
			ROS_INFO("world frame from head frame 0,0,0 = %f, %f, %f", worldZeroPointConsoleMsg.x, worldZeroPointConsoleMsg.y, worldZeroPointConsoleMsg.z);
			ROS_INFO("color values RGB = %f, %f, %f", finalConsoleMsg.r, finalConsoleMsg.g, finalConsoleMsg.b);
			ROS_INFO("This was target number %d", numTargets_);
			numTargets_++;
			//geometry_msgs::Point rvizMarker;
			//rvizMarker.x = finalConsoleMsg.x;
			//rvizMarker.y = finalConsoleMsg.y;
			//rvizMarker.z = finalConsoleMsg.z;
			
			previousConsoleMsg_.x = averageMsg.x;
			previousConsoleMsg_.y = averageMsg.y;
			previousConsoleMsg_.z = averageMsg.z;
			previousConsoleMsg_.b = averageMsg.b;
			previousConsoleMsg_.g = averageMsg.g;
			previousConsoleMsg_.r = averageMsg.r;
		}
		else 
		{
			ROS_WARN("could not find good XYZ value");
			consolePub_.publish(finalConsoleMsg);
			result = false;
		}
	}
	else
	{
		ROS_INFO("console message not published, disparity was 0");
		result = false;
	}
	//ROS_INFO("examine images");
	
	leftImageReady_ = false;
	rightImageReady_ = false;
	rightImageAnalyzed_ = true;
	leftImageAnalyzed_ = true;
	distanceCalculated_ = true;
	newCloudReceived_ = false;
	alreadyPrintedWhiteboxCenter_ = false;
	centerHead(); // wait for this to return
	return result;
}

void laserCallback(const sensor_msgs::LaserScan msg)
{
	scanNum_++;
	laserHeader_ = msg.header;
	laserHeaderFrameID_ = laserHeader_.frame_id;
	laserAngle_min_ = msg.angle_min;
	laserAngle_max_ = msg.angle_max;
	laserAngle_increment_ = msg.angle_increment;
	laserRanges_ = msg.ranges;
}

double getLaserRange(double deltaY)
{		
	std::vector<float> localRanges = laserRanges_;
	// deltaY is positive to the left, so if we have a positive value,
	// we want a laser index that is less than localRanges.size() / 2
	double angleToLight = atan2(deltaY, localRanges[ ((int) localRanges.size() / 2) ]);
	int laserIndex = int ((localRanges.size() / 2) - (angleToLight / laserAngle_increment_));
	double rangeToLight = localRanges[laserIndex];
	ROS_INFO("distance from center = %f, laser index = %d, range = %f", deltaY, laserIndex - ((int) (localRanges.size() / 2)), rangeToLight);
	return rangeToLight;
}

void moveHead(double deltaHeight)
{
	std::vector<float> localRanges = laserRanges_;
	double angleToMove = atan2(deltaHeight, localRanges[ ((int) localRanges.size() / 2) ]);
	// pubish message to move head
}

void centerHead()
{
	// publish message to put head back to center
}
		
bool writeToFile(std::string filename, Mat imageFrame)
{
   if (imageFrame.empty()) return false;
   vector<int> compression_params; //vector that stores the compression parameters of the image
   compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //specify the compression technique
   compression_params.push_back(98); //specify the compression quality
   bool bSuccess = imwrite(filename, imageFrame, compression_params); //write the image to file
   if (bSuccess) 
   {
      //ROS_INFO("image written to file, filename: %s", filename.c_str());
      return true;
   }
   else ROS_ERROR("error in writing image to file");
   return false;
}

};
	
int main(int argc, char** argv)
{
	ros::init(argc, argv, "consoleAnalysis");
	ros::NodeHandle nh;
	consoleAnalysis cA(nh);
	cv::namedWindow("left_result", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("right_result", CV_WINDOW_AUTOSIZE);
	cv::startWindowThread();

	/*
	// test files
	std::string filename = "/home/dbarry/Dropbox/catkin/src/srobotics/data/qual1_images/console0Left.jpg";
	Mat image = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	cA.analyzeImage(image, LEFT);
	
	filename = "/home/dbarry/Dropbox/catkin/src/srobotics/data/qual1_images/console0Right.jpg";
	image = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	cA.analyzeImage(image, RIGHT);
	*/
	//delay = ros::Duration(10.0);
	
	// laser data
	/*
	std::vector<double> goodRanges;
	std::vector<double> goodAngles;
   long lastScanNum = cA.scanNum_;
	while (nh.ok())
	{   
		std::cout << "Enter any key to analyze scan data: " << std::endl;
		std::string input = "";
		getline(std::cin, input);
		while (cA.scanNum_ == lastScanNum)
		{
			ros::spinOnce(); // wait for a new scan
		}
	
		lastScanNum = cA.scanNum_;
		std::vector<float> localRanges = cA.laserRanges_;
		std::cout << std::endl << std::endl;
		std::cout << "scan number: " << lastScanNum << " has " << localRanges.size() << " values" << std::endl;
		std::cout << "angle min, max, increment = " << cA.laserAngle_min_ << ", " 
			<< cA.laserAngle_max_ << ", " << cA.laserAngle_increment_ << std::endl;
		//for (unsigned int i=0; i < localRanges.size(); i++)
		//{
		//	if (std::isfinite(localRanges[i]))
		//	{
		//		goodRanges.push_back(localRanges[i]);
		//		goodAngles.push_back(cA.laserAngle_min_ + (i * cA.laserAngle_increment_));
		//	}
		//}
		//for (int i = (localRanges.size() / 2) - 200; i < (localRanges.size() / 2) + 200; i++)
		//{
		//	std::cout << "range = " << localRanges[i] << std::endl;
		//}
		
		double deltaY = 0.55;
		// deltaY is positive to the left, so if we have a positive value,
		// we want a laser index that is less than localRanges.size() / 2
		double angleToLight = atan2(deltaY, localRanges[ ((int) localRanges.size() / 2) ]);
		int laserIndex = int ((localRanges.size() / 2) - (angleToLight / cA.laserAngle_increment_));
		double rangeToLight = localRanges[laserIndex];
		ROS_INFO("laser index = %d, range = %f", laserIndex - ((int) (localRanges.size() / 2)), rangeToLight); 
		 
   }

	cv::destroyWindow("left_result");
	cv::destroyWindow("right_result");
	return EXIT_SUCCESS;
	*/
	
	
	ros::Rate r(1);
	while (nh.ok())
	{
		if (cA.leftImageReceived_) {
			if (!cA.leftImageAnalyzed_) {
				if (cA.analyzeImage(LEFT)) 
				{
					cA.leftImageAnalyzed_ = true;
					cA.readyForRightImage_ = true;
				}
				else {
					//ROS_INFO("analyze left image returned false");
					cA.leftImageAnalyzed_ = false;
					cA.leftImageReceived_ = false;
					cA.readyForLeftImage_ = true;
					cA.readyForRightImage_ = false;
				}					
			}
			else {
				if (cA.rightImageReceived_) {
					if (!cA.rightImageAnalyzed_) {
						if (cA.analyzeImage(RIGHT)) {
							cA.rightImageAnalyzed_ = true;
							cA.readyForCloud_ = true;
						}
						else {
							ROS_INFO("analyze right image returned false");
							cA.leftImageReceived_ = false;
							cA.rightImageReceived_ = false;
							cA.leftImageAnalyzed_ = false;
							cA.rightImageAnalyzed_ = false;
							cA.readyForLeftImage_ = true;
							cA.readyForRightImage_ = false;
						}
					}	
					else {				
						if (cA.newCloudReceived_) {
							if (cA.calculateDistance()) {
								//ROS_INFO("calculateDistance returned true \n");
							}
							else {
								//ROS_INFO("calculateDistance returned false \n");
							}
							cA.leftImageReceived_ = false;
							cA.rightImageReceived_ = false;
							cA.leftImageAnalyzed_ = false;
							cA.rightImageAnalyzed_ = false;
							cA.newCloudReceived_ = false;
							cA.readyForRightImage_ = false;
							cA.readyForCloud_ = false;
							cA.readyForLeftImage_ = true;
						}
						//else ROS_INFO("waiting for new cloud to be received");
					}
				}					
				else {
					//ROS_INFO("waiting for new right image to be received");
				}
			}
		}
		else {
			//ROS_INFO("waiting for new left image to be received");
			cA.readyForLeftImage_ = true;
		}
		//cA.markerPoints_.header.stamp = ros::Time::now();
		cA.markerPub_.publish(cA.markerPoints_);
		//r.sleep();
		ros::spinOnce();
		//ROS_INFO("hit any key to go on \n");
		//cv::waitKey(0);
	}
	cv::destroyWindow("left_result");
	cv::destroyWindow("right_result");
	return EXIT_SUCCESS;
}	
	
