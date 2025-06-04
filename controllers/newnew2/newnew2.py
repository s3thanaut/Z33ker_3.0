from controller import Robot
import numpy as np
import csv
import os
import cv2
import time

robot = Robot()
TIME_STEP = 64

# Camera setup
camera = robot.getDevice("cam")
camera.enable(TIME_STEP)
camera_width = camera.getWidth()
camera_height = camera.getHeight()

controller_dir = os.path.dirname(os.path.abspath(__file__))
LEARNED_PATH_FILE = os.path.join(controller_dir, 'learned_path.csv')

# Devices
motor_names = ['motorL', 'motorR']
sensor_names = ['senF', 'senL1', 'senL2', 'senR1', 'senR2']

motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

sensors = {name: robot.getDevice(name) for name in sensor_names}
for sensor in sensors.values():
    sensor.enable(TIME_STEP)

enL = robot.getDevice('enL')
enR = robot.getDevice('enR')
enL.enable(TIME_STEP)
enR.enable(TIME_STEP)

imu = robot.getDevice('imu')
imu.enable(TIME_STEP)

# Constants
WHEEL_RADIUS = 0.02
TRACK_WIDTH = 0.11
BASE_SPEED = 5.0
FAST_SPEED = 15.0
KP = 0.005
KD = 0.003
KI = 0.001
LOOKAHEAD_DIST = 0.2
ANGULAR_K = 5.0

# HSV thresholds for red
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 100, 100])
UPPER_RED_2 = np.array([180, 255, 255])

ROI_Y_START = int(camera_height * 0.8)
ROI_Y_END = camera_height
ROI_X_START = 0
ROI_X_END = camera_width
MIN_RED_PIXEL_COUNT = 50

# State
lap_triggered = False
red_line_crossed = False
line_cross_start_time = None
path_x, path_y = [], []
current_x = current_y = current_theta = 0.0
prev_enc_L = prev_enc_R = 0.0
path_index = 0
previous_error = integ = 0


def detect_red_line(image_data):
    np_img = np.frombuffer(image_data, np.uint8).reshape((camera_height, camera_width, 4))
    bgr_img = np_img[:, :, :3]
    roi = bgr_img[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixel_count = cv2.countNonZero(red_mask)
    return red_pixel_count > MIN_RED_PIXEL_COUNT


def update_odometry(x, y, theta, prev_L, prev_R):
    cur_L = enL.getValue()
    cur_R = enR.getValue()
    dL = (cur_L - prev_L) * WHEEL_RADIUS
    dR = (cur_R - prev_R) * WHEEL_RADIUS
    dC = (dL + dR) / 2
    theta = imu.getRollPitchYaw()[2]
    x += dC * np.cos(theta)
    y += dC * np.sin(theta)
    return x, y, theta, cur_L, cur_R


def save_path(px, py):
    with open(LEARNED_PATH_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for x, y in zip(px, py):
            writer.writerow([x, y])
    print("Path saved.")


def load_path():
    px, py = [], []
    with open(LEARNED_PATH_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            px.append(float(row[0]))
            py.append(float(row[1]))
    return px, py


# Main control loop
while robot.step(TIME_STEP) != -1:
    current_x, current_y, current_theta, prev_enc_L, prev_enc_R = update_odometry(
        current_x, current_y, current_theta, prev_enc_L, prev_enc_R
    )

    camera_image_data = camera.getImage()
    red_line_detected = detect_red_line(camera_image_data) if camera_image_data else False

    # Delay path switching after crossing red line
    if red_line_detected and not red_line_crossed and not lap_triggered:
        print("Red line detected. Crossing...")
        red_line_crossed = True
        line_cross_start_time = time.time()

    if red_line_crossed and not lap_triggered:
        if time.time() - line_cross_start_time > 0.5:  # wait for 0.5s to ensure robot crosses
            print("Switching to path following.")
            save_path(path_x, path_y)
            path_x, path_y = load_path()
            lap_triggered = True
            path_index = 0
        else:
            motors[0].setVelocity(BASE_SPEED)
            motors[1].setVelocity(BASE_SPEED)
            continue

    if not lap_triggered:
        e = (sensors['senR1'].getValue() + sensors['senR2'].getValue()*1.5) - (sensors['senL1'].getValue() + sensors['senL2'].getValue()*1.5)
        d = e - previous_error
        integ += e
        correction = KP * e + KD * d + KI * integ
        previous_error = e

        left_speed = BASE_SPEED - correction
        right_speed = BASE_SPEED + correction

        if sensors['senF'].getValue() > 800:
            left_speed = -BASE_SPEED
            right_speed = -BASE_SPEED / 2

        motors[0].setVelocity(min(20.0, max(0.0, left_speed)))
        motors[1].setVelocity(min(20.0, max(0.0, right_speed)))

        path_x.append(current_x)
        path_y.append(current_y)
    else:
        lookahead_found = False
        for i in range(path_index, len(path_x)):
            dx = path_x[i] - current_x
            dy = path_y[i] - current_y
            dist = np.hypot(dx, dy)
            if dist >= LOOKAHEAD_DIST:
                target_x = path_x[i]
                target_y = path_y[i]
                path_index = i
                lookahead_found = True
                break

        if not lookahead_found:
            print("Reached end of path. Moving forward to red line before stopping.")
            line_detect_time = time.time()
            while robot.step(TIME_STEP) != -1:
                camera_image_data = camera.getImage()
                red_line_detected = detect_red_line(camera_image_data) if camera_image_data else False
                
                if not red_line_detected and time.time() - line_detect_time < 5:
                    # Move forward at FAST_SPEED
                    motors[0].setVelocity(FAST_SPEED)
                    motors[1].setVelocity(FAST_SPEED)
                else:
                    # Stop the robot after 3 seconds
                    motors[0].setVelocity(0)
                    motors[1].setVelocity(0)
                    break
            break  # exit main loop after stopping

        dx = target_x - current_x
        dy = target_y - current_y
        target_angle = np.arctan2(dy, dx)
        angle_error = np.arctan2(np.sin(target_angle - current_theta), np.cos(target_angle - current_theta))

        correction = ANGULAR_K * angle_error
        left_speed = FAST_SPEED - correction
        right_speed = FAST_SPEED + correction

        motors[0].setVelocity(min(20.0, max(-20.0, left_speed)))
        motors[1].setVelocity(min(20.0, max(-20.0, right_speed)))
