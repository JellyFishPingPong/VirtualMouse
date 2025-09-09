#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

import pyautogui
wCam = 1280
yCam = 720

import time

open_start_time = 0
close_start_time = 0
ok_start_time = 0
sensitivity = 1.2

drag_start_time = 0

pyautogui.PAUSE = 0


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=wCam)
    parser.add_argument("--height", help='cap height', type=int, default=yCam)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    MOVE = 0
    SCROLL = 1
    DRAG = 2

    SCROLL_UP = 0
    SCROLL_DOWN = 1
    scroll_direction = 0
    prev_scroll_y = 0
    max_scroll_speed = 20  # Adjust to your preference

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    screen_width, screen_height = pyautogui.size()
    margin = 10

    # Define the fixed box dimensions and position (in camera frame pixels)
    box_x1 = int(cap_width * 0.2)
    box_y1 = int(cap_height * 0.2)
    box_width = int(cap_width * 0.5)
    box_height = int(cap_height * 0.5)
    box_x2 = box_x1 + box_width
    box_y2 = box_y1 + box_height
    fixed_box_color = (0, 255, 0)  # Green color for the box
    fixed_box_thickness = 2

    camera_on = True
    controls_on = False
    toggle_key = ord('c')  # You can change this key

    # Define the hand sign
    open_sign = 0
    close_sign = 1
    pointer_sign = 2
    ok_sign = 3
    slowScrollUp_sign = 4
    slowScrollDown_sign = 5
    thumbs_up_sign = 6
    rock_sign = 7
    click_sign = 8
    right_click_sign = 9

    global open_start_time, close_start_time, ok_start_time, drag_start_time

    cursor_mode = MOVE
    swipe_triggered = False
    can_click = True
    can_click_r = True
    can_change_direction = True

    previous_sign = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end, 'c' to toggle camera) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == toggle_key:
            camera_on = not camera_on
            print(f"Camera {'ON' if camera_on else 'OFF'}")
        number, mode = select_mode(key, mode)

        if camera_on:
            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Draw the fixed box on the debug image
            cv.rectangle(debug_image, (box_x1, box_y1), (box_x2, box_y2),
                         fixed_box_color, fixed_box_thickness)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Hand sign classification
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Finger gesture classification
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    if finger_gesture_id != 1:  # reset swipe availability when no swiping anymore
                        swipe_triggered = False

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    # <editor-fold desc="Hand gesture with no conditions">
                    # OPEN SIGN
                    if hand_sign_id == open_sign:
                        if not controls_on:
                            if open_start_time == 0:
                                open_start_time = time.time()
                            elif time.time() - open_start_time >= 1:
                                print("controls on")
                                controls_on = True
                                open_start_time = 0  # Reset the timer
                    else:
                        open_start_time = 0

                    # OKAY SIGN
                    if hand_sign_id == ok_sign:
                        if ok_start_time == 0:
                            ok_start_time = time.time()
                        elif time.time() - ok_start_time >= 1.5:
                            print("Alt+F4 triggered after 1 second!")
                            pyautogui.hotkey('alt', 'f4')
                            ok_start_time = 0  # Reset the timer
                    else:
                        ok_start_time = 0

                    if hand_sign_id == thumbs_up_sign and previous_sign != thumbs_up_sign:
                        if scroll_direction == SCROLL_UP:
                            scroll_direction = SCROLL_DOWN
                        else:
                            scroll_direction = SCROLL_UP

                    if hand_sign_id == rock_sign and previous_sign != rock_sign:
                        print("change mode")
                        if cursor_mode != MOVE:
                            cursor_mode = MOVE
                        else:
                            cursor_mode = SCROLL

                    if cursor_mode == SCROLL:
                        if hand_sign_id == pointer_sign:
                            # Get index finger coordinates in camera frame pixels
                            index_finger_y = int(landmark_list[8][1])

                            # Calculate the difference between current and previous y-coordinate
                            y_difference = index_finger_y - prev_scroll_y

                            # Check if scrolling up
                            if scroll_direction == SCROLL_UP and y_difference > 0:
                                scroll_amount = abs(y_difference)
                                pyautogui.scroll(scroll_amount)

                            # Check if scrolling down
                            elif scroll_direction == SCROLL_DOWN and y_difference < 0:
                                scroll_amount = abs(y_difference)
                                pyautogui.scroll(-scroll_amount)

                            # Update the previous y-coordinate
                            prev_scroll_y = index_finger_y

                        if hand_sign_id == slowScrollUp_sign:
                            pyautogui.scroll(20)

                        if hand_sign_id == slowScrollDown_sign:
                            pyautogui.scroll(-20)
                    else:
                        # CLOSE SIGN
                        if hand_sign_id == close_sign:
                            if close_start_time == 0:
                                close_start_time = time.time()
                            elif time.time() - close_start_time >= 1.5:
                                print("off control triggered")
                                controls_on = False
                                close_start_time = 0  # Reset the timer
                        else:
                            close_start_time = 0
                    # </editor-fold>

                    point_history.append(landmark_list[8])
                    # <editor-fold desc="Gesture that require control to be on">
                    if controls_on:

                        if hand_sign_id == rock_sign:
                            if finger_gesture_id == 1 and not swipe_triggered and controls_on:
                                pyautogui.hotkey('alt', 'esc')
                                print("escaped")
                                swipe_triggered = True

                        # CURSOR MOVE MODE
                        if cursor_mode == MOVE:

                            if hand_sign_id == pointer_sign:
                                # Get index finger coordinates in camera frame pixels
                                index_finger_x = int(landmark_list[8][0])
                                index_finger_y = int(landmark_list[8][1])

                                # Check if the index finger is inside the fixed box
                                if box_x1 < index_finger_x < box_x2 and box_y1 < index_finger_y < box_y2 and finger_gesture_id != 0:
                                    # Map the finger position within the box to screen coordinates
                                    screen_x = int(((index_finger_x - box_x1) / box_width) * screen_width * sensitivity)
                                    screen_y = int(
                                        ((index_finger_y - box_y1) / box_height) * screen_height * sensitivity)

                                    # Apply margin to the screen coordinates
                                    screen_x = max(min(screen_x, screen_width - margin), margin)
                                    screen_y = max(min(screen_y, screen_height - margin), margin)
                                    pyautogui.moveTo(screen_x, screen_y, 0)

                            if hand_sign_id == click_sign:
                                if can_click and controls_on:
                                    pyautogui.click()
                                    print("click")
                                    can_click = False
                                if cursor_mode != DRAG:
                                    if drag_start_time == 0:
                                        drag_start_time = time.time()
                                        print("Click hold started...")
                                    else:
                                        hold_duration = time.time() - drag_start_time
                                        if hold_duration >= 1:
                                            print("Entering drag mode...")
                                            # cursor_mode = DRAG
                                        else:
                                            print(f"Holding for {hold_duration:.2f} seconds...")
                            else:
                                can_click = True
                                drag_start_time = 0  # Reset the timer

                            if hand_sign_id == right_click_sign:
                                if can_click_r and controls_on:
                                    pyautogui.click(button="right")
                                    print("right click")
                                    can_click_r = False
                            else:
                                can_click_r = True

                        elif cursor_mode == SCROLL:
                            pass

                        elif cursor_mode == DRAG:
                            pass


                    else:
                        point_history.append([0, 0])
                    # </editor-fold>

                    previous_sign = hand_sign_id

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
            else:
                clear_timer()
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)
        else:
            # If camera is off, display a black screen or a message
            off_screen = np.zeros_like(debug_image)
            cv.putText(off_screen, "Camera OFF", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv.imshow('Hand Gesture Recognition', off_screen)
            clear_timer()
    cap.release()
    cv.destroyAllWindows()


def clear_timer(exception = ""):
    global open_start_time, close_start_time, ok_start_time, drag_start_time
    if exception != "open":
        open_start_time = 0
    if exception != "close":
        close_start_time = 0
    if exception != "ok":
        ok_start_time = 0
    if exception != "drag":
        drag_start_time = 0


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
