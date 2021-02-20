# To get the object detection working with Realsense:

- cd ~/catkin_ws/src
- git clone https://github.com/jatan12/ROS_Yolo2D.git
- cd ~/catkin_ws
- catkin_make
- roslaunch realsense2_camera rs_camera.launch

### Open up a new terminal, then:
- conda activate
- rosrun ROS_Yolo2D final_yolo.py

### NB! Refer to the setup guide in order to install the necessary packages to get started

https://docs.google.com/document/d/13qFxHlV3pAJqZAZfvzfcouMZbMDDu6xn5O5VixYNci8/edit
