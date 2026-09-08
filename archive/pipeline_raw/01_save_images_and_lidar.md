# Raw pipeline reference: 01_save_images_and_lidar.ipynb

```python
import numpy as np
import threading
# -*- coding: utf-8 -*-
from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.qcar import QLabsQCar
import time
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from qvl.system import QLabsSystem
from ultralytics import YOLO

from time import sleep
from PIL import Image as im
import torch
import copy
```

```python
os.system('cls')

#Communications with qlabs

qlabs = QuanserInteractiveLabs()
cv2.startWindowThread()

print("Connecting to QLabs...")

for i in range(1000):
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
    else:
        break

print("Connected")

# Use hSystem to set the tutorial title in the upper left of the qlabs window 
hSystem = QLabsSystem(qlabs)
hSystem.set_title_string('QCar Tutorial')


print("\n\n---QCar---")
#spawning the QCar with radians



```

```python
hQCar0 = QLabsQCar(qlabs)
hQCar0.actorNumber = 0
hQCar1 = QLabsQCar(qlabs)
hQCar1.actorNumber = 4
n=30
out_data = []
out_data1 = []
target = 0.135
c=0

model_detec = YOLO('C:\\Users\\l_daryel\\Downloads\\best.pt')
model_detec.to('cuda')
FOCAL_LENGTH = 320      # example value in pixels
CAR_HEIGHT = 0.19       # assumed target height in meters
out_data2 = []
flag = True

while flag:
    x, camera_image = hQCar0.get_image(camera=hQCar0.CAMERA_CSI_FRONT)
    results = model_detec.track(camera_image, persist=True)
    if results:
        result = results[0]
        for box in result.boxes:
            conf = box.conf[0]
            if conf >= 0.75:
                flag = False

while hQCar0.ping() and c<=30:
    begin = time.time()
    x, camera_image = hQCar0.get_image(camera=hQCar0.CAMERA_CSI_FRONT)
    if camera_image is not None:
        camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        out_data.append(camera_image_rgb)
    x, angles, distance = hQCar0.get_lidar(samplePoints=2000)
    if angles is not None:
        out_data1.append(list(zip(angles, distance)))

    finish = time.time()

    time_taken = finish - begin
    sleep_time = target - time_taken

    if sleep_time > 0:
        sleep(sleep_time)
    print(time_taken)
    c+=1

dir=f"C:/Users/l_daryel/Downloads/Paper2/Images/Scenario{n}"
os.makedirs(dir, exist_ok=True)

for i in range(len(out_data)):
    if len(out_data) != 0:
        filename = os.path.join(dir, f"image{i}.png")
        front = im.fromarray(out_data[i], mode='RGB')
        front.save(filename)

dir = f"C:/Users/l_daryel/Downloads/Paper2/Lidar/Scenario{n}"
os.makedirs(dir, exist_ok=True)

for i in range(len(out_data1)):
    filename = os.path.join(dir, f"lidar{i}.txt")
    np.savetxt(filename, out_data1[i], fmt="%.6f", delimiter=" ")

```

```python
n=30
images = []
for i in range(4, 30):
    filename = f"C:/Users/l_daryel/Downloads/Paper2/Images/Physical/Scenario{n}/image{i}.png"
    # Load file
    sc = im.open(filename).convert('RGB')
    loaded_array_bgr = np.array(sc)
    images.append(loaded_array_bgr)
total_images = np.array(images)
```

```python
all_images = copy.deepcopy(total_images)
```

```python
model_detec = YOLO('C:\\Users\\l_daryel\\Downloads\\best1 (1).pt')
model_detec.to('cuda')
FOCAL_LENGTH = 320      # example value in pixels
CAR_HEIGHT = 1.5       # assumed target height in meters
out_data2 = []


for image in all_images:

    results = model_detec.track(image, persist=True)
    if results:
        current_time = time.time()
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.id[0]) if box.id is not None else 'None'

            # Bounding box center + size
            Lp = y2 - y1
            bbox_center_x = (x2 + x1) / 2
            image_height, image_width = image.shape[:2]
            image_center_x = image_width / 2

            if conf >= 0.2 and Lp > 0:
                # --- Angle estimation ---
                angle = (bbox_center_x - image_center_x) / FOCAL_LENGTH
                angle_degrees = math.degrees(angle)

                # --- Distance estimation (pinhole model) ---
                est_distance = (CAR_HEIGHT * FOCAL_LENGTH) / Lp
                # Draw on frame
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image,
                            f'ID:{class_id}, Dist:{est_distance:.2f}m, Angle: {angle_degrees:.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
        out_data2.append(image.copy())

        # Show frame
        cv2.imshow('Front', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue
cv2.destroyWindow('Front')
dir=f"C:/Users/l_daryel/Downloads/Paper2/Detections/Physical/Scenario{n}"
os.makedirs(dir, exist_ok=True)

for i in range(len(out_data2)):
    if len(out_data2) != 0:
        filename = os.path.join(dir, f"image{i}.png")
        front = im.fromarray(out_data2[i], mode='RGB')
        front.save(filename)
```