# Uncomment when using the realsense camera
import pyrealsense2.pyrealsense2 as rs  # For (most) Linux and Macs
import math

# import pyrealsense2 as rs # For Windows
import numpy as np
import logging
import time
import datetime
import drone_lib
# import fg_camera_sim
import cv2
import imutils
import random
import logging
import traceback
import sys
import os
import glob
import shutil
from pathlib import Path

# visdrone stuff
output_layers = None
visdrone_net = None
visdrone_classes = []
CONF_THRESH, NMS_THRESH = 0.05, 0.5

global drone
global log
global frame_w, frame_h
# global object_identified
log = None  # logger instance
GRIPPER_OPEN = 1087
GRIPPER_CLOSED = 1940
gripper_state = GRIPPER_CLOSED  # assume gripper is closed by default
# IMG_SNAPSHOT_PATH = '/dev/drone_data/mission_data/cam_pex003'
IMG_WRITE_RATE = 10  # write every 10 frames to disk...

# Various mission states:
# We start out in "seek" mode, if we think we have a target, we move to "confirm" mode,
# If target not confirmed, we move back to "seek" mode.
# Once a target is confirmed, we move to "target" mode.
# After positioning to target and calculating a drop point, we move to "deliver" mode
# After delivering package, we move to RTL to return home.
MISSION_MODE_SEEK = 0
MISSION_MODE_CONFIRM = 1
MISSION_MODE_TARGET = 2
MISSION_MODE_DELIVER = 4
MISSION_MODE_RTL = 8

# Tracks the state of the mission
mission_mode = MISSION_MODE_SEEK

# Number of frames in a row we need to confirm a suspected target
REQUIRED_SIGHT_COUNT = 1  # must get 60 target sightings in a row to be sure of actual target

# Violet target
# COLOR_RANGE_MIN = (110, 100, 75)
# COLOR_RANGE_MAX = (160, 255, 255)

# Blue (ish) target
# COLOR_RANGE_MIN = (80, 50, 50)
# COLOR_RANGE_MAX = (105, 255, 255)

# Smallest object radius to consider (in pixels)
MIN_OBJ_RADIUS = 10

UPDATE_RATE = 1  # How many frames do we wait to execute on.

TARGET_RADIUS_MULTI = 1.7  # 1.5 x the radius of the target is considered a "good" landing if drone is inside of it.

# Font for use with the information window
font = cv2.FONT_HERSHEY_SIMPLEX

# variables
drone = None
counter = 0
direction1 = "unknown"
direction2 = "unknown"
inside_circle = False

# tracks number of attempts to re-acquire a target (if lost)
target_locate_attempts = 0

# Holds the size of a potential target's radius
target_circle_radius = 0

# info related to last (potential) target sighting
last_obj_lon = None
last_obj_lat = None
last_obj_alt = None
last_obj_heading = None
last_point = None  # center point in pixels

# Configure realsense camera stream
pipeline = rs.pipeline()
config = rs.config()

# x,y center for 640x480 camera resolution.
FRAME_HORIZONTAL_CENTER = int(320)
FRAME_VERTICAL_CENTER = int(240)
FRAME_HEIGHT = int(480)
FRAME_WIDTH = int(640)

rnd_background = np.random.randint(0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3)).astype('uint8')

total_track_misses = 0
TRACKER_MISSES_MAX = 35
confirmed_object_tracking = False

tracker = None
DEFAULT_TRACKER_TYPE = 'CSRT'
cv_version = cv2.__version__


def create_tracker(tracker_type='CSRT'):
    global tracker

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
        # tracker = cv2.TrackerMIL_create()
    return tracker


def get_new_lat_lon(from_lat, from_lon, heading, distance):
    earthRadius = 6378.1
    heading = math.radians(last_obj_heading)

    lat1 = math.radians(from_lat)
    lon1 = math.radians(from_lon)

    lat2 = math.asin(math.sin(lat1) * math.cos((distance*0.001) / earthRadius) +
                     math.cos(lat1) * math.sin((distance*0.001) / earthRadius) * math.cos(heading))

    lon2 = lon1 + math.atan2(math.sin(heading) * math.sin((distance*0.001) / earthRadius) * math.cos(lat1),
                             math.cos((distance*0.001) / earthRadius) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    print(lat2)
    print(lon2)

    return lat2, lon2


def calc_new_location_to_target(from_lat, from_lon, heading, distance):
    from geopy import distance
    from geopy import Point

    # given: current latitude, current longitude,
    #        heading = bearing in degrees,
    #        distance from current location (in meters)

    origin = Point(from_lat, from_lon)
    destination = distance.distance(
        kilometers=(float(distance * .001))).destination(origin, heading)

    return destination.latitude, destination.longitude


def load_visdrone_network():
    global visdrone_net, output_layers, visdrone_classes

    in_weights = 'yolo_visdrone/yolov4-tiny-custom_last.weights'
    in_config = 'yolo_visdrone/yolov4-tiny-custom.cfg'
    name_file = 'yolo_visdrone/custom.names'
    # in_weights = 'yolov4-tiny-custom_last.weights'
    # in_config = 'yolov4-tiny-custom.cfg'
    # name_file = 'custom.names'

    """
    load names
    """
    with open(name_file, "r") as f:
        visdrone_classes = [line.strip() for line in f.readlines()]

    """
    Load the network
    """
    visdrone_net = cv2.dnn.readNetFromDarknet(in_config, in_weights)
    visdrone_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    visdrone_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layers = visdrone_net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in visdrone_net.getUnconnectedOutLayers()]


def confirm_obj_in_bbox(frame, bbox):
    try:
        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            return False
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cr = int(max(w, h) / 2)

        # blank_image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        blank_image = rnd_background.copy()  # np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8(100))
        cropped = frame[cy - cr:cy + cr, cx - cr:cx + cr]

        blank_image[y:y + cropped.shape[0], x:x + cropped.shape[1]] = cropped
        cv2.imshow("cropped", blank_image)

        center, confidence, (x, y), radius, frm_display, bbox = check_for_initial_target(blank_image)

        if confidence is not None \
                and confidence > .2:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def check_for_initial_target(img=None):
    if img is None:
        img = get_cur_frame()

    my_color = (20, 20, 230)
    b_boxes, scores, class_ids = detect_object(img, visdrone_net)

    scores_kept = []
    b_boxes_kept = []

    # only consider people here...
    for index in range(0, len(scores)):
        if (visdrone_classes[class_ids[index]] == "pedestrian"
                or visdrone_classes[class_ids[index]] == "people")\
                or visdrone_classes[class_ids[index]] == "car":
            # if visdrone_classes[class_ids[index]] == "car":
            scores_kept.append(scores[index])
            b_boxes_kept.append(b_boxes[index])

    if len(scores_kept) >= 1:
        max_confidence = max(scores_kept)
        max_index = scores_kept.index(max_confidence)

        x, y, w, h = b_boxes_kept[max_index]
        cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 230), 2)
        cv2.putText(img, f"{visdrone_classes[class_ids[max_index]]}, {scores_kept[max_index]}"
                    , (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, my_color, 2)

        x_center = x + int(w / 2)
        y_center = y + (int(h / 2))

        return (x_center, y_center), max_confidence, (x, y), max([h / 2, w / 2]), img, b_boxes_kept[max_index]
    else:
        return (0, 0), None, (0, 0), None, img, (0, 0, 0, 0)


def track_with_confirm(img):
    global total_track_misses, confirmed_object_tracking

    # Here, we will use an object track to ensure we're
    #       tracking the very same object we identified earlier.
    center, confidence, (x, y), radius, frm_display, bbox = track_object(img.copy())
    if confidence is not None:

        if confirm_obj_in_bbox(img.copy(), bbox):
            total_track_misses = 0
            confirmed_object_tracking = True
        else:
            total_track_misses += 1
    else:
        # Tracking failure
        total_track_misses += 1

    if total_track_misses >= TRACKER_MISSES_MAX:
        # Tracking failure
        confirmed_object_tracking = False
        confidence = None
    else:
        cv2.putText(frm_display, "Tracking...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)

    return center, confidence, (x, y), radius, frm_display, bbox


def track_object(img):
    # Here, we will use an object tracker to ensure we're
    #       tracking the very same object we identified earlier.

    ok, box = tracker.update(img)
    if ok:
        bbox = tuple(int(val) for val in box)
        x, y, w, h = bbox
        x_center = bbox[0] + int(bbox[2] / 2)
        y_center = bbox[1] + (int(bbox[3] / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        return (x_center, y_center), 1.0, (x, y), max([h / 2, w / 2]), img, bbox
    else:
        return (0, 0), None, (0, 0), None, img, (0, 0, 0, 0)


def detect_object(img, net):
    cv2.putText(img, 'detecting...', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (192, 192), swapRB=False, crop=False)

    # blob = cv2.dnn.blobFromImage(
    #    cv2.resize(img, (416, 416)),
    #    0.007843, (416, 416), 127.5)

    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:

        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                center_x, center_y, w, h = \
                    (detection[0:4] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # hold our final results here...
    final_bboxes = []
    final_scores = []
    final_class_ids = []

    if len(b_boxes) > 0:
        # Perform non maximum suppression for the bounding boxes
        # to filter overlapping and low confidence bounding boxes.
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
        for index in indices:
            final_bboxes.append(b_boxes[index])
            final_class_ids.append(class_ids[index])
            final_scores.append(confidences[index])

    return final_bboxes, final_scores, final_class_ids


def start_camera_stream():
    # comment our when not testing in sim...

    global pipeline, config
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    image_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    image_intrinsics = image_profile.get_intrinsics()
    frame_w, frame_h = image_intrinsics.width, image_intrinsics.height


def get_cur_frame(attempts=5, flip_v=False):
    # Wait for a coherent pair of frames: depth and color
    tries = 0

    # This will capture the frames from the simulator.
    # If using an actual camera, comment out the two lines of
    # code below and replace with code that returns a single frame
    # from your camera.
    # image = fg_camera_sim.get_cur_frame()
    # return cv2.resize(image, (int(FRAME_HORIZONTAL_CENTER * 2), int(FRAME_VERTICAL_CENTER * 2)))

    # Code below can be used with the realsense camera...
    while tries <= attempts:
        try:
            frames = pipeline.wait_for_frames()
            rgb_frame = frames.get_color_frame()
            rgb_frame = np.asanyarray(rgb_frame.get_data())

            if flip_v:
                rgb_frame = cv2.flip(rgb_frame, 0)
            return rgb_frame
        except Exception as e:
            print(e)

        tries += 1


def set_object_to_track(frame, bbox, bbox_margin=25):
    # On some platforms, the tracker reset doesn't work,
    # so we need to create a new instance here.
    tracker = create_tracker(DEFAULT_TRACKER_TYPE)

    if bbox_margin <= 0:
        tracker.init(frame, bbox)
    else:
        # center original bbox within a larger, square bbox
        x, y, w, h = bbox

        ## get the center and the radius
        cx = x + w // 2
        cy = y + h // 2
        cr = max(w, h) // 2

        r = cr + bbox_margin
        new_bbox = [cx - r, cy - r, r * 2, r * 2]
        x, y, w, h = new_bbox
        tracker.init(frame, (x, y, w, h))


def release_grip(seconds=2):
    sec = 1

    while sec <= seconds:
        override_gripper_state(GRIPPER_OPEN)
        time.sleep(1)
        sec += 1


def override_gripper_state(state=GRIPPER_CLOSED):
    global gripper_state
    gripper_state = state
    drone.channels.overrides['7'] = gripper_state


def backup_prev_experiment(path):
    if os.path.exists(path):
        if len(glob.glob(f'{path}/*')) > 0:
            time_stamp = time.time()
            shutil.move(os.path.normpath(path),
                        os.path.normpath(f'{path}_{time_stamp}'))

    Path(path).mkdir(parents=True, exist_ok=True)


def clear_path(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)


def conduct_mission():

    logging.info("Searching for target...")

    target_sightings = 0
    global counter, mission_mode, last_point, last_obj_lon, \
        last_obj_lat, last_obj_alt, \
        last_obj_heading, target_circle_radius

    # init video
    start_camera_stream()
    load_visdrone_network()
    object_identified = False

    movements = 0
    withinX = 0
    withinY = 0

    while drone.armed:
        # Start timer
        location = drone.location.global_relative_frame
        last_lon = location.lon
        last_lat = location.lat
        last_alt = location.alt
        last_heading = drone.heading

        timer = cv2.getTickCount()
        frame = get_cur_frame()
        frm_display = frame.copy()

        if drone.mode == "RTL":
            mission_mode = MISSION_MODE_RTL
            logging.info("RTL mode activated. Mission ended.")
            break

        if not object_identified:
            if drone.mode != "AUTO":
                drone_lib.change_device_mode(drone, "AUTO")
            center, confidence, (x, y), radius, frm_display, bbox \
                = check_for_initial_target(frm_display)
            if confidence is not None \
                    and confidence > .2:
                # Initialize tracker with first frame and bounding box
                # bbox needs: xb,yb,wb,hb
                object_identified = True
                set_object_to_track(frame, bbox)
                if drone.mode != "GUIDED":
                    drone_lib.change_device_mode(drone, "GUIDED")
                # while drone.location.global_relative_frame.lat != last_lat and drone.location.global_relative_frame.lon != last_lon:
                drone_lib.goto_point(drone, last_lat, last_lon, 1, last_alt, log=log)
                ogbbox = bbox
        else:
            # if drone.mode == "LOITER":
            #     break
            if drone.mode != "GUIDED":
                drone_lib.change_device_mode(drone, "GUIDED")
            center, confidence, (x, y), radius, frm_display, bbox \
                = track_with_confirm(frm_display)
            if withinX < 5 and withinY < 5:
                if (center[0]) < (FRAME_WIDTH / 2):
                    if (center[0]) < (FRAME_WIDTH / 2)-15:
                        drone_lib.small_move_left(drone)
                        movements = movements + 1
                    else:
                        withinX = withinX + 1
                else:
                    if (center[0]) > (FRAME_WIDTH / 2):
                        if (center[0]) > (FRAME_WIDTH / 2)+15:
                            drone_lib.small_move_right(drone)
                            movements = movements + 1
                        else:
                            withinX = withinX + 1
                if (center[1]) > (FRAME_HEIGHT / 2):
                    if (center[1]) > (FRAME_HEIGHT / 2) + 15:
                        drone_lib.small_move_back(drone)
                        movements = movements + 1
                    else:
                        withinY = withinY + 1
                else:
                    if (center[1]) < (FRAME_HEIGHT / 2) - 15:
                        drone_lib.small_move_forward(drone)
                        movements = movements + 1
                    else:
                        withinY = withinY + 1
            else:
                target_sightings = target_sightings + 1

            last_obj_lon = last_lon
            last_obj_lat = last_lat
            last_obj_alt = last_alt
            last_obj_heading = last_heading
            if not confidence:
                cv2.putText(frm_display,
                            "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
                target_sightings = 0
                movements = 0
                withinX = 0
                withinY = 0
                object_identified = False

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # Display FPS on frame
        cv2.putText(frm_display, "FPS : " + str(int(fps)),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Real-time Detect", frm_display)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if target_sightings > 60:
            determine_drone_actions((last_lat, last_lon), frm_display, target_sightings, ogbbox)
            break

    # while True:
    # drone_lib.goto_point(drone, drone.location.global_relative_frame.lat, drone.location.global_relative_frame.lon, speed=.2, alt=8)
    time.sleep(2)
    drone_lib.goto_point(drone,  drone.location.global_relative_frame.lat, drone.location.global_relative_frame.lon, speed=.1, alt=2.5)
    release_grip(2)

    drone_lib.change_device_mode(drone, "RTL")

def get_hypotenuse(img, bbox):
    # ret, thresh = cv2.threshold(img, x, y, cv2.THRESH_BINARY_INV)
    # num_pixels = cv2.countNonZero(thresh)
    # num_pixels = ((x + radius) - x) * ((y + radius) - y)

    num_pixels = bbox[2] * bbox[3]
    print("num_pixels")
    print(num_pixels)

    heightRat = (drone.location.global_relative_frame.alt*.01)
    dist_ratio1 = (drone.location.global_relative_frame.alt / 7.646)* heightRat

    # dist_ratio = .16795
    pix_ratio = .01065
    hypo = num_pixels * (pix_ratio * dist_ratio1)
    return hypo


def get_ground_distance(height, hypotenuse):
    import math

    # Assuming we know the distance to object from the air
    # (the hypotenuse), we can calculate the ground distance
    # by using the simple formula of:
    # d^2 = hypotenuse^2 - height^2

    return math.sqrt(hypotenuse ** 2 - height ** 2)


def determine_drone_actions(last_point, frame, target_sightings, bbox):
    if target_sightings >= 60:
        hypo = get_hypotenuse(frame, bbox)
        print("hypo")
        print(hypo)
        # hypo = 1.5*drone.location.global_relative_frame.alt
        # distance = get_ground_distance(drone.location.global_relative_frame.alt, hypo)

        print("distance")
        print(hypo)
        # new_lat, new_lon = calc_new_location_to_target(last_obj_lat, last_obj_lon, last_obj_heading, distance)
        new_lat, new_lon = get_new_lat_lon(last_obj_lat, last_obj_lon, last_obj_heading, hypo)
        drone_lib.goto_point(drone, new_lat, new_lon, speed=.5, alt=5)
        # drone_lib.change_device_mode(drone, "LOITER")

#

if __name__ == '__main__':

    # init video
    start_camera_stream()
    load_visdrone_network()
    # object_identified = False

    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Setup a log file for recording important activities during our session.
    log_file = time.strftime("TEAM_NAME_PEX03_%Y%m%d-%H%M%S") + ".log"

    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    log = logging.getLogger(__name__)

    log.info("PEX 03 start.")

    # Connect to the autopilot
    drone = drone_lib.connect_device("127.0.0.1:14550", log=log)
    # drone = drone_lib.connect_device("/dev/ttyACM0", baud=115200, log=log)

    # Create a message listener using the decorator.
    print(f"Finder above ground: {drone.rangefinder.distance}")

    # Test latch - ensure open/close.
    release_grip(2)

    # If the autopilot has no mission, terminate program
    drone.commands.download()
    time.sleep(1)

    log.info("Looking for mission to execute...")
    if drone.commands.count < 1:
        log.info("No mission to execute.")
        exit()

    # Arm the drone.
    drone_lib.arm_device(drone, log=log)

    # takeoff and climb 45 meters
    drone_lib.device_takeoff(drone, 20, log=log)

    try:
        # start mission
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        log.info("backing up old images...")

        # Backup any previous images and create new empty folder for current experiment.
        # backup_prev_experiment(IMG_SNAPSHOT_PATH)

        # Now, look for target...
        conduct_mission()

        # Mission is over; disarm and disconnect.
        log.info("Disarming device...")
        drone.armed = False
        drone.close()
        log.info("End of demonstration.")
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise
