# Uncomment when using the realsense camera
# import pyrealsense2.pyrealsense2 as rs # For (most) Linux and Macs
# import pyrealsense2 as rs # For Windows

import logging
import time
import drone_lib
import fg_camera_sim
import cv2
import imutils
import random
import logging
import traceback
import sys

log = None  # logger instance

# Various mission states:
# We start out in "seek" mode, if we think we have a target, we move to "confirm" mode,
# If target not confirmed, we move back to "seek" mode.
# Once a target is confirmed, we move to "target" mode.
# After descending to about 10ft above target, we move to the final state, "land".
MISSION_MODE_SEEK = 0
MISSION_MODE_TARGET = 1
MISSION_MODE_CONFIRM = 2
MISSION_MODE_LAND = 3
MISSION_MODE_LOITER = 4

# x,y center for 640x480 camera resolution.
FRAME_HORIZONTAL_CENTER = 320.0
FRAME_VERTICAL_CENTER = 240.0

# Number of frames in a row we need to confirm a suspected target
REQUIRED_SIGHT_COUNT = 1  # must get 60 target sightings in a row to be sure of actual target

# Min HSV values
# TODO: you must determine your own minimum for color range by
#    analysing your target's color.
#  0,0,0 will not work very well for you.
COLOR_RANGE_MIN = (85, 67, 105)

# Max HSV values
# TODO: you must determine your own maximum for color range by
#    analysing your target's color.
#  0,0,0 will not work very well for you.
#  See comments in check_for_initial_target() related to basic thresholding on a range of HSV
COLOR_RANGE_MAX = (255, 91, 255)

# Smallest object radius to consider (in pixels)
MIN_OBJ_RADIUS = 10

UPDATE_RATE = 1  # How many frames do we wait to execute on.

TARGET_RADIUS_MULTI = 2  # 2 x the radius of the target is considered a "good" landing if drone is inside of it.

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

# Tracks the state of the mission
mission_mode = MISSION_MODE_SEEK

# info related to last (potential) target sighting
last_obj_lon = None
last_obj_lat = None
last_obj_alt = None
last_obj_heading = None
last_point = None  # center point in pixels


# Uncomment below when using actual realsense camera
# Configure realsense camera stream
# pipeline = rs.pipeline()
# config = rs.config()


def start_camera_stream():
    logging.info("configuring rgb stream.")
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    logging.info("Starting camera streams...")
    # profile = pipeline.start(config)


def get_cur_frame(attempts=5):
    # Wait for a coherent pair of frames: depth and color
    tries = 0

    # This will capture the frames from the simulator.
    # If using an actual camera, comment out the two lines of
    # code below and replace with code that returns a single frame
    # from your camera.
    image = fg_camera_sim.get_cur_frame()
    # print("pass through")
    return cv2.resize(image, (int(FRAME_HORIZONTAL_CENTER * 2), int(FRAME_VERTICAL_CENTER * 2)))

    # Code below can be used with the realsense camera...
    # while tries <= attempts:
    #    try:
    #        frames = pipeline.wait_for_frames()
    #        rgb_frame = frames.get_color_frame()
    #        rgb_frame = np.asanyarray(rgb_frame.get_data())
    #        return rgb_frame
    #    except Exception:
    #        print(Exception)
    #
    #    tries+=1


def check_for_initial_target():
    global inside_circle

    # get current frame from the camera
    frame = get_cur_frame()

    # Apply a Gaussian blur; this is a widely used effect in computer vision,
    # typically to reduce image noise and (slightly) reduce hard lines.
    # You can find more details here:
    # https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    # TODO: YOU COMPLETE the line of code below:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Get hue, saturation, value.
    # Hue, saturation, and value are the main color properties that
    # allow us to distinguish between different colors.
    # ** When you refer to HUE, you are referring to pure color,
    #    or the visible spectrum of basic colors that can be seen in a rainbow.
    # ** Color SATURATION is the purity and intensity of a color as displayed
    #    in an image. The higher the saturation of a color, the more vivid and intense it is.
    # ** Color VALUE refers to the relative lightness or darkness of a color.
    # Note: since the hue channel models the color type, it is
    # very useful in image processing tasks that need to SEGMENT
    # objects based on color.
    # Go here to learn more: https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/

    # TODO: YOU COMPLETE the line of code below:
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #print(hsv)
    # Perform basic thresholding on a range of HSV.
    # This page explains the details: https://docs.opencv.org/master/da/d97/tutorial_threshold_inRange.html
    # This site offers more insight for thesholding HSV: https://blog.socratesk.com/blog/2018/08/16/opencv-hsv-selector
    # USAGE: threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # Use wide range to account for variations of light, dirt, and surface imperfections...
    # especially as the drone gets closer to the target on approach.

    # TODO: YOU COMPLETE the line of code below:
    color_threshold = cv2.inRange(hsv, COLOR_RANGE_MIN, COLOR_RANGE_MAX)

    # Now, perform some basic morphological operations to enhance the shapes present in the image.
    # Morphological operations are a set of operations that process images based on shapes;
    # they apply a structuring element to an input image and generate an output image.
    # Now, the most basic morphological operations are: Erosion and Dilation.
    # They have a wide array of uses, i.e. :
    # 1. Removing noise
    # 2. Isolation of individual elements and joining disparate elements in an image.
    # 3. Finding of intensity bumps or holes in an image
    #  This link provides useful information on the subject:
    #  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    # First, the basic idea of erosion is just like soil erosion;
    # it erodes away the boundaries of foreground object
    # (always try to keep foreground in white).
    # It computes a local minimum over the area of given kernel.
    # As a kernel B is scanned over the image, we compute the minimal pixel
    # value overlapped by B and replace the image pixel under the anchor point
    # with that minimal value.

    # TODO: YOU COMPLETE the line of code below:
    color_threshold = cv2.erode(color_threshold, None, iterations=2)

    # Next, the dilate operation consists of convolving an image A with some kernel B,
    # which can have any shape or size, usually a square or circle.
    # The result is the "fattening" of our circle in this case.

    # TODO: YOU COMPLETE the line of code below:
    color_threshold = cv2.dilate(color_threshold, None, iterations=2)

    # By this point we essentially have a binary image (i.e. basic black & white features)

    # Contour in an image is an outline on the objects present in the image.
    # The significance of the objects depend on the requirement and threshold we choose.
    # In this case we're looking for a clean circle after we've filtered, eroded, and dilated.
    # See: http://datahacker.rs/006-opencv-projects-how-to-detect-contours-and-match-shapes-in-an-image-in-python/
    # Note that finding contours is like finding white objects from a black background.
    # So remember, the object(s) to be found should be white and background should be black.
    # Go here for details: https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html

    # TODO: YOU COMPLETE the line of code below:
    found_contours = cv2.findContours(color_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Now, we want to remove any noise (i.e. any contours like dots and other specks caused by intense light reflection)
    # and focus on the largest contour from the processed image.
    # This discussion provides more details:
    # https://stackoverflow.com/questions/55062579/opencv-how-to-clear-contour-from-noise-and-false-positives/55249329

    found_contours = imutils.grab_contours(found_contours)

    center = None
    radius = None
    x = y = None

    if len(found_contours) > 0:
        # Let's consider the contour with the largest area.
        circle = max(found_contours, key=cv2.contourArea)

        # Get x,y and radius of the circle that surrounds the object we're examining.
        # Essentially, we want to find the circumcircle of an object
        # using the function cv.minEnclosingCircle().
        # The result is a circle which completely covers the object with minimum area.
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        # Note that this technique has a flaw in that ANY shape of sufficient area that matches
        # the color we're looking for can be mistaken for our target; the enclosing
        # circle is what we're looking at, even if the object we encircled wasn't a circle
        # or any other specific shape
        ((x, y), radius) = cv2.minEnclosingCircle(circle)

        # The function cv2.moments() gives a dictionary of all moment values calculated.
        # From the moments, you can extract useful data like area, centroid etc.
        # See wikipedia for more info: https://en.wikipedia.org/wiki/Image_moment
        middle = cv2.moments(circle)

        # To get center:
        cx = int(middle['m10'] / middle['m00'])
        cy = int(middle['m01'] / middle['m00'])

        # TODO: YOU COMPLETE the code below:
        # TODO MAY NOT WORK
        center = (cx, cy)

    return center, radius, (x, y), frame


def determine_drone_actions(target_point, frame, target_sightings):
    # Based on the (potential) target and the current mode we're in,
    # determine what actions the drone should take (if any) at this point in time.
    # We'll examine the current snapshot of time (the frame) to decide what to do...

    # NOTE: you are not required to use anything specific here.
    #       You may change whatever you like here to make the code
    #       more efficient or simply perform better.

    global mission_mode, target_locate_attempts
    global direction1, direction2
    global target_circle_radius, inside_circle
    global last_obj_lon, last_obj_lat, last_obj_alt, last_obj_heading

    y_movement = 0.0
    x_movement = 0.0

    # Now, lets calculate our drone's actions according to what we have found...
    if target_point is not None:

        # TODO: determine dx and dy here (drone's position relative to the target's center)
        #   Note that this is in pixels.
        #   Check may not work, especially the frame part
        # dx = float(target_point's x position)- frame's horizontal center
        center, radius, (x, y), frame2 = check_for_initial_target()
        dx = float(x) - FRAME_HORIZONTAL_CENTER
        # dy = frame's vertical center -float(target_point's y position)
        dy = FRAME_VERTICAL_CENTER - float(y)

        logging.info(f"Anticipated change in position towards target: dx={dx}, dy={dy}")

        # Draw a line between most-recent center point of target and
        # the drone's position (i.e. the center of the frame).
        cv2.line(frame, target_point,
                 (int(FRAME_HORIZONTAL_CENTER),
                  int(FRAME_VERTICAL_CENTER)),
                 (0, 0, 255), 5)

        # Check to see if we're inside our safe zone relative to target...
        if (int(target_point[0]) - FRAME_HORIZONTAL_CENTER) ** 2 \
                + (int(target_point[1]) - FRAME_VERTICAL_CENTER) ** 2 \
                < target_circle_radius ** 2:

            inside_circle = True
        else:
            inside_circle = False

        logging.info(f"Inside target zone: {inside_circle}.")

    # Display whether we're inside the circle or not (informational only)
    cv2.putText(frame, f"Inside zone: {inside_circle}",
                (10, 120), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Determine if we need to assess a potential target further...
    if mission_mode != MISSION_MODE_TARGET \
            and mission_mode != MISSION_MODE_CONFIRM \
            and target_point is not None:

        # If we have not officially confirmed a target, let's attempt to do so
        # by switching to "confirm" mode; we need to confirm the target.

        # Time to pause and take a closer look at the target...
        # You can stop/pause the current flight plan by switching out of AUTO mode (e.g. into GUIDED mode).
        # If you switch back to AUTO mode the mission will either restart at the beginning or
        # resume at the current waypoint.
        # The behaviour depends on the value of the MIS_RESTART parameter.
        # (We want to be sure that the drone is configured with MIS_RESTART = 0)
        mission_mode = MISSION_MODE_CONFIRM  # we will confirm the target next time this function is called.

        logging.info(f"Need to confirm target.")

        # Drone's internal mission is temporarily
        # suspended until we can confirm target.

        # TODO: YOU ADD required line of code below:
        #    switch over to guided mode so that we can control the drone's movements
        #    (don't forget to pass the log object).
        # SET to drone.mode
        drone_lib.change_device_mode(drone, "GUIDED", log=log)

    else:
        if mission_mode == MISSION_MODE_CONFIRM \
                and target_point is not None:

            logging.info(f"Confirming target...")

            cv2.putText(frame, "Doing double-take...",
                        (10, 400), font, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            if target_sightings == 0:
                logging.info(f"No target confirmation.")
                mission_mode = MISSION_MODE_SEEK
                last_obj_lon = None
                last_obj_lat = None
                last_obj_alt = None
                last_obj_heading = None

                # TODO: YOU COMPLETE the line of code below:
                #    switch back to auto mode so that the drone can resume the internal flight plan
                #    (don't forget to pass the log object).
                drone_lib.change_device_mode(drone, "AUTO", 10, log)

            else:
                # Here, if the target is spotted the required
                # number of times, we consider it "confirmed".
                if target_sightings >= REQUIRED_SIGHT_COUNT:
                    logging.info(f"Target confirmed.")

                    # Begin the landing process by switching over to
                    # "target" mode; while in this mode we attempt
                    # to center within the target during landing.
                    mission_mode = MISSION_MODE_TARGET

                    # First, re-position over the point where
                    # where the target was first spotted, and
                    # increase altitude by 5 meters to better fit the
                    # target in the incoming images.
                    logging.info(f"Positioning towards target...")

                    # TODO: YOU COMPLETE the 2 lines of code below:
                    #   goto point where
                    #   the target was originally spotted (don't forget to pass the log object)
                    #   hint: ast_obj_lat, last_obj_lon, drone.airspeed, ast_obj_alt+5, last_obj_heading
                    #   1. move to point here
                    #   2. perform yaw to face in right direction here.
                    drone_lib.goto_point(drone, last_obj_lat, last_obj_lon, drone.airspeed, last_obj_alt + 5, log=log)
                    drone_lib.condition_yaw(drone, last_obj_heading)

    # Execute drone commands...
    if mission_mode == MISSION_MODE_TARGET:
        if drone.location.global_relative_frame.alt <= 2:           #change to 2 meters

            # time to land...
            logging.info("Time to Release...")

            drone_lib.change_device_mode(drone, "LOITER", log=log)                      #change to loiter then release then climb 3 meters then RTL

            #call release function
            print("Released")

            #make drone climb a couple meters
            drone_lib.change_device_mode(drone, "GUIDED", log=log)
            drone_lib.move_local(drone, 0, 0, -.5, duration=2, log=log)

            # TODO: YOU COMPLETE the line of code below:
            #    land the drone (don't forget to pass the log object)
            # drone_lib.device_land(drone, log=log)

            # cv2.putText(frame, "Landing...", (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            drone_lib.change_device_mode(drone, "RTL", log=log)

            return  # Mission is over (hopefully, all is well).
        else:
            # Adjust position relative to target
            if target_point is not None:
                # Determine how big our x,y adjustments are
                # based on how close we are to the target.

                # Note: movements velocity in meters per second (m/s)
                mov_inc = 1  # start out with full meter per second.

                if drone.location.global_relative_frame.alt <= 20:
                    mov_inc = 0.5  # make smaller adjustments as we get closer to target

                if drone.location.global_relative_frame.alt < 11:
                    mov_inc = 0.2  # make smaller adjustments as we get closer to target

                # TODO: YOU COMPLETE the lines of code below:
                #    determine y_movement and x_movement
                #    (also, rate of decent - could be fixed to .5 m/s)
### got to change from move_local to something else

                if dx < 0:  # left
                    # do what?  negative direction...
                    x_movement = -mov_inc
                    # if(last_obj_alt < 18):
                    #     x_movement = -.2
                    # print("left")
                if dx > 0:  # right
                    # do what?  positive direction...
                    x_movement = mov_inc
                    # if (last_obj_alt < 18):
                    #     x_movement = .2
                    # print("right")
                if dy < 0:  # back
                    # do what?  positive direction...
                    y_movement = mov_inc
                    # if (last_obj_alt < 18):
                    #     y_movement = .2
                    # print("back")
                if dy > 0:  # forward
                    # do what?  negative direction...
                    y_movement = -mov_inc
                    # if (last_obj_alt < 18):
                    #     y_movement = -.2
                    # print("forward")
                if abs(dx) < 10:  # if we are within 8 pixels, no need to make adjustment
                    x_movement = 0.0
                    direction1 = "Horizontal Center!"
                if abs(dy) < 10:  # if we are within 8 pixels, no need to make adjustment
                    y_movement = 0.0
                    direction2 = "Vertical Center!"

                # log movements...
                logging.info("Targeting... determined changes in velocities: X: "
                             + str(x_movement) + ", Y:"
                             + str(y_movement) + ", Z:"
                             + str(0.5) + ".")
                print("Targeting... determined changes in velocities: X: "
                             + str(x_movement) + ", Y:"
                             + str(y_movement) + ", Z:"
                             + str(0.5) + ".")
                cv2.putText(frame, "Targeting...", (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Have the drone execute movements.
                # TODO: YOU COMPLETE the line of code below: move the drone a tiny bit towards the target's
                #   center point by executing on x_movement and y_movement (don't forget to pass the log object).
                #   Note that move_local expects velocities, not actual x,y,z positions
                #   figure out line of code below to get drone to make minor adjustment to current X,Y position
                #   while descending at .5 m/s

                drone_lib.move_local(drone, x_movement, y_movement, .5, duration=1, log=log)

            else:  # We lost the target...

                # NOTE: below is code that attempts to re-locate a target if it was lost.
                #  This method is hardly the best approach, but it works
                #  With that said, you may opt to do anything you want here to
                #  improve upon the approach.

                logging.info("Target lost.")

                # Ty to recover the target by moving to the
                # last sighting and scanning for the target again.
                target_locate_attempts += 1

                # We will only make a limited number of attempts to
                # re-acquire the target before giving up and
                # resuming the drone's internal mission.
                if target_locate_attempts <= 20:
                    logging.info("Re-acquiring target...")
                    cv2.putText(frame, "Re-acquiring target...",
                                (10, 400), font, 1,
                                (255, 0, 0), 2, cv2.LINE_AA)

                    # Move to point of original sighting.
                    # TODO: YOU COMPLETE the 2 lines of code below:
                    #   goto point where
                    #   the target was originally spotted (don't forget to pass the log object)
                    #   1. move to point here
                    #   2. perform RANDOM yaw here for a different vantage point than before
                    drone_lib.goto_point(drone, last_obj_lat, last_obj_lon, drone.airspeed, last_obj_alt, log)
                    drone_lib.condition_yaw(drone, random.randint(0,359))

                else:
                    # if we failed to re-locate the target,
                    # then continue on with the drone's internal mission...
                    logging.info("Discarding previous target, continue looking for another...")
                    mission_mode = MISSION_MODE_SEEK
                    target_locate_attempts = 0
                    last_obj_lon = None
                    last_obj_lat = None
                    last_obj_alt = None
                    last_obj_heading = None

                    # TODO: YOU ADD required line of code below:
                    #    switch over to whatever mode you need to
                    #    so as to resume the drone's internal mission
                    #    (don't forget to pass the log object)
                    drone_lib.change_device_mode(drone, "AUTO", log = log)


def search_for_target():
    # Here, we will loop until we find a target and land,
    # or until the drone's mission  completes (and we land).
    logging.info("Searching for target...")

    target_sightings = 0
    global counter, last_point, last_obj_lon, \
        last_obj_lat, last_obj_alt, \
        last_obj_heading, target_circle_radius

    logging.info("Starting camera feed...")
    start_camera_stream()

    while drone.armed:  # While the drone's mission is executing...

        # take a snapshot of current location
        location = drone.location.global_relative_frame
        last_lon = location.lon
        last_lat = location.lat
        last_alt = location.alt
        last_heading = drone.heading

        # look for a target in current frame
        center, radius, (x, y), frame = check_for_initial_target()

        if center is not None \
                and radius > MIN_OBJ_RADIUS:  # if we found something that might be the target...

            logging.info(f"(Potential) target acquired @"
                         f"({center[0], center[1]}) with radius {radius}.")

            target_sightings += 1

            # We're looking for an object that meets both
            # color and size (i.e. radius) requirements.

            # Note that to determine how many pixels across the radius should appear,
            # we need to know the actual size of our target radius and then
            # scale it to what it would appear to be at max of 50ft (max distance from target).
            # This can be different for different cameras, depending on focal length and sensor size.
            last_point = center

            if mission_mode == MISSION_MODE_SEEK:
                logging.info(f"Locking in on lat {last_lat}, lon {last_lon}, "
                             f"alt {last_alt}, heading {last_heading}.")

                last_obj_lon = last_lon
                last_obj_lat = last_lat
                last_obj_alt = last_alt
                last_obj_heading = last_heading

            # Draw tight (red) circle around identified target
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

            # Calculate and draw our acceptable safe zone (2x larger than original target).
            target_circle_radius = radius * TARGET_RADIUS_MULTI
            cv2.circle(frame, (int(x), int(y)), int(target_circle_radius), (255, 255, 0), 2)

            # Draw point at center of target.
            cv2.circle(frame, center, 5, (0, 255, 255), -1)

        else:
            # We have no target in the current frame.
            logging.info("No target found; continuing search...")
            cv2.putText(frame, "Scanning for target...", (10, 400), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            target_sightings = 0  # reset target sighting
            last_point = None

        # Time to adjust drone's position?
        if (counter % UPDATE_RATE) == 0 \
                or mission_mode != MISSION_MODE_SEEK:
            # determine drone's next actions (if any)
            determine_drone_actions(last_point, frame, target_sightings)

        # Display information in windowed frame:
        cv2.putText(frame, direction1, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, direction2, (10, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw (blue) marker in center of frame that indicates the
        # drone's relative position to the target
        # (assuming camera is centered under the drone).
        cv2.circle(frame,
                   (int(FRAME_HORIZONTAL_CENTER), int(FRAME_VERTICAL_CENTER)),
                   10, (255, 0, 0), -1)

        # Now, show stats for informational purposes only
        cv2.imshow("Real-time Detect", frame)
        time.sleep(.1)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if mission_mode == MISSION_MODE_LAND:
            return  # mission is over.

        counter += 1


def main():
    global drone
    global log

    # Setup a log file for recording important activities during our session.
    log_file = time.strftime("%Y%m%d-%H%M%S") + ".log"

    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    log = logging.getLogger(__name__)

    log.info("PEX 02 start.")

    # Connect to the autopilot
    drone = drone_lib.connect_device("127.0.0.1:14550", log=log)
    # drone = connect_device("COM5",baud=57600, log=log)

    # If the autopilot has no mission, terminate program
    drone.commands.download()
    time.sleep(1)

    log.info("Looking for mission to execute...")
    if drone.commands.count < 1:
        log.info("No mission to execute.")
        return

    # Arm the drone.
    drone_lib.arm_device(drone, log=log)

    # takeoff and climb 45 meters
    drone_lib.device_takeoff(drone, 20, log=log)

    try:
        # start mission
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        # Now, look for target...
        search_for_target()

        # Mission is over; disarm and disconnect.
        log.info("Disarming device...")
        drone.armed = False
        drone.close()
        log.info("End of demonstration.")
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise


main()
