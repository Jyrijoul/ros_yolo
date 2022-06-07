#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image

# Initialize the global variables before getting the values from the parameter server.
# Show the output image?
show_image = True
# Don't refresh without key presses?
freeze_detection = False
# Publish bounding box data even when no objects found?
publish_empty = False

import sys

# Due to dependency hell, this is needed:
sys.path.extend(
    [
        "/home/jyri/catkin_ws/src/ros_yolo/scripts",
    ]
)
# print(sys.path)


# Necessary Imports
import time
import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    plot_one_box,
    strip_optimizer,
    set_logging,
)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from matplotlib import pyplot as plt

ros_image = 0


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def loadimg(img):
    img_size = 640
    cap = None
    path = None
    img0 = img
    img = letterbox(img0, new_shape=img_size)[0]
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return path, img, img0, cap


def detect(img):

    time1 = time.time()

    global ros_image
    cudnn.benchmark = True
    dataset = loadimg(img)
    # print(dataset[3])
    # plt.imshow(dataset[2][:, :, ::-1])
    names = model.module.names if hasattr(model, "module") else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors=[[0,255,0]]
    augment = "store_true"
    conf_thres = 0.3
    iou_thres = 0.45
    classes = (0, 1, 2, 3, 5, 7)
    agnostic_nms = "store_true"
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != "cpu" else None  # run once
    path = dataset[0]
    img = dataset[1]
    im0s = dataset[2]
    vid_cap = dataset[3]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    time2 = time.time()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )

    view_img = 1
    save_txt = 1
    save_conf = "store_true"
    time3 = time.time()

    bb_to_publish = []

    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, "", im0s
        s += "%gx%g " % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None:
            # print(det)
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += "%g %ss, " % (n, names[int(c)])  # add to string
                # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (
                        (cls, conf, *xywh) if save_conf else (cls, *xywh)
                    )  # label format
                if view_img:  # Add bbox to image
                    label = "%s %.2f" % (names[int(cls)], conf)
                    plot_one_box(
                        xyxy, im0, label=label, color=[0, 255, 0], line_thickness=3
                    )
                    x = xyxy
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    # print("XYXY:", c1, c2)
                    bb_to_publish.extend([c1[0], c1[1], c2[0], c2[1]])

    # Package the bounding boxes into one list, 4 values constituting one box.
    if bb_to_publish or publish_empty:
        publish_bounding_boxes(bb_to_publish)
    # else:
    #     # Also publish an empty list if not detecting anything.
    #     publish_bounding_boxes([])

    time4 = time.time()
    # print("************")
    # print("2-1", time2 - time1)
    # print("3-2", time3 - time2)
    # print("4-3", time4 - time3)
    # print("total", time4 - time1)
    # rospy.loginfo("Rate = " + str(round(1 / (time4 - time1), 3)) + " Hz.")
    if show_image:

        # out_img = im0[:, :, [2, 1, 0]]
        out_img = im0
        # out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        cv2.imshow("YOLOV5", out_img)

        if freeze_detection:
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                # Not sure whether the next line is necessary...
                rospy.loginfo("Pressed 'q' to shut down.")
                rospy.signal_shutdown("Pressed 'q' to shut down.")
        else:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                # Not sure whether the next line is necessary...
                rospy.loginfo("Pressed 'q' to shut down.")
                rospy.signal_shutdown("Pressed 'q' to shut down.")
    #### Create Image ####
    publish_image(im0)


def image_callback(image):
    global ros_image
    ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
        image.height, image.width, -1
    )
    with torch.no_grad():
        detect(ros_image)


def publish_image(imgdata):
    # TODO: just read in image encoding.
    if simulation:
        image_temp = Image(encoding="bgr8")
    else:
        image_temp = Image(encoding="rgb8")
    header = Header(stamp=rospy.Time.now())
    header.frame_id = output_frame
    image_temp.header = header
    image_temp.height = imgdata.shape[0]
    image_temp.width = imgdata.shape[1]
    # image_temp.data = np.array(imgdata).tobytes()
    # image_temp.step = image_temp.width*3
    # Inspiration from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py.
    contig = np.ascontiguousarray(imgdata)
    image_temp.data = contig.tobytes()
    image_temp.step = contig.strides[0]
    image_pub.publish(image_temp)


def publish_bounding_boxes(xyxy):
    message = Int32MultiArray()
    message.data = xyxy
    bb_pub.publish(message)


if __name__ == "__main__":
    set_logging()
    device = ""
    device = select_device(device)
    half = device.type != "cpu"  # half precision only supported on CUDA
    weights = "yolov5s.pt"
    imgsz = 640
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    rospy.init_node("ros_yolo")

    # Show the output image?
    show_image = rospy.get_param("~show_image", default=True)
    # Don't refresh without key presses?
    freeze_detection = rospy.get_param("~freeze_detection", default=False)
    # Publish bounding box data even when no objects found?
    publish_empty = rospy.get_param("~publish_empty", default=False)
    # Output coordinate frame
    output_frame = rospy.get_param("~output_frame", default="camera_link")
    # Simulation?
    simulation = rospy.get_param("~simulation", default=True)

    # The following topics will be remapped.
    camera_image_topic = "camera_image"
    yolo_image_out_topic = "yolo_image_out"
    yolo_bounding_box_out_topic = "yolo_bounding_box_out"

    image_pub = rospy.Publisher(yolo_image_out_topic, Image, queue_size=1)
    rospy.loginfo(f"Started publishing to topic {image_pub.resolved_name}.")
    bb_pub = rospy.Publisher(yolo_bounding_box_out_topic, Int32MultiArray, queue_size=1)
    rospy.loginfo(f"Started publishing to topic {bb_pub.resolved_name}.")
    image_sub = rospy.Subscriber(
        camera_image_topic, Image, image_callback, queue_size=1, buff_size=52428800
    )
    rospy.loginfo(f"Started subscribing to topic {image_sub.resolved_name}.")
    rospy.spin()
