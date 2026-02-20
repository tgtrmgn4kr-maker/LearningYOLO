import cv2
import sys
import math
import numpy as np
from ultralytics import YOLO

def calc_angle(s, e, w): 
    s2e = np.array(s) - np.array(e) # vector of shoulder to elbow  
    w2e = np.array(w) - np.array(e) # vector of wrist to elbow  

    inner_product = np.dot(s2e, w2e)

    length_s2e = np.linalg.norm(s2e)
    length_w2e = np.linalg.norm(w2e)

    angle_deg = np.rad2deg(np.arccos(inner_product / (length_s2e * length_w2e)))
    return angle_deg

def draw_circle_label(frame, keypoint, point_id):
    x, y = int(keypoint[0]), int(keypoint[1])
    cv2.circle(frame,
               (x, y),
               5,
               (0, 255, 0),
               -1
               )
    
    cv2.putText(frame,
                str(point_id),
                (x+5, y-5),
                0,
                1,
                (255, 0, 0),
                2
                )
    
def check_arm(frame, person, arm_index, is_down, count, position):
    shoulder = person[arm_index[0]]
    elbow = person[arm_index[1]]
    wrist = person[arm_index[2]]

    if (shoulder is not None) and (elbow is not None) and (wrist is not None):
        angle = calc_angle(shoulder, elbow, wrist)

        cv2.putText(frame,
                    f"{int(angle)}",
                    (position[0], position[1]+50),
                    0,
                    1,
                    (255, 255, 255),
                    2
                    )
        
        if angle <= UP_THRESHOLD and not is_down:
            is_down = True

        elif angle >= DOWN_THRESHOLD and is_down:
            count += 1
            is_down = False

    cv2.putText(frame,
                f"{count}",
                position,
                0,
                3,
                (0, 0, 255),
                5
                )
    
    return is_down, count

cap = cv2.VideoCapture(0)
model = YOLO("yolo11n-pose.pt")

UP_THRESHOLD = 90
DOWN_THRESHOLD = 150
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

right_is_down = False
right_count = 0
left_is_down = False
left_count = 0

if not cap.isOpened():
    print("Camera is not opened")
    sys.exit(1)

while True:
    ret, frame = cap.read()

    results = model.track(frame, persist=True)

    if results[0].keypoints is not None:

        kp_data = results[0].keypoints.xy.numpy()

        for person in kp_data:
            right_is_down, right_count = check_arm(frame,
                                                   person,
                                                   (6, 8, 10),
                                                   right_is_down,
                                                   right_count,
                                                   (50, 100)
                                                   )
            
            left_is_down, left_count = check_arm(frame,
                                                 person,
                                                 (5, 7, 9),
                                                 left_is_down,
                                                 left_count,
                                                 (FRAME_WIDTH-150, 100))
            
            for i in range(5, 11):
                if person[i][0] > 0:
                    draw_circle_label(frame,
                                      person[i],
                                      i
                                      )
                    
    cv2.imshow("Dumbbell", frame)

    if cv2.waitKey(1) & 0b1111_1111 == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()