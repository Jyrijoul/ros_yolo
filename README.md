# ROS YOLO object detection package

This work is a fork of https://github.com/jatan12/ROS_Yolo2D.git.

## To get the object detection working with RealSense:

- Refer to the setup guide _(setup.pdf)_ in order to install the necessary packages.
- Use ```roslaunch realsense2_camera rs_camera.launch``` or some other image source.
    - The launch file has the following arguments:
        - ```camera_image``` - the input image topic
        - ```yolo_image_out``` - the output image topic (with the bounding box)
        - ```yolo_bounding_box_out``` - the output topic for bounding box coordinates
- Then, use ```roslaunch ros_yolo yolo.launch``` to perform object detection.
    - Due to a dependency issue, line 18 of _final_yolo.py_ will need to be changed accordingly.
