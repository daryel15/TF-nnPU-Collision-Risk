# Raw pipeline reference: 02_feature_extraction_camera_lidar.ipynb

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
from sklearn.cluster import DBSCAN
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
all_features = []
points = []
n = 17
nmax=81
############################ Lidar
for i in range(4, nmax):
    filename = f"Lidar/Scenario{n}/lidar{i}.txt"
    # Load file
    sc = np.loadtxt(filename)
    points.append(sc)
total_points = np.array(points)
########################### Images
images = []
for i in range(4, nmax):
    filename = f"Images/Scenario{n}/image{i}.png"
    # Load file
    sc = im.open(filename).convert('RGB')
    loaded_array_bgr = np.array(sc)
    images.append(loaded_array_bgr)
total_images = np.array(images)
########################### Detetions
detects = []
for i in range(4, nmax):
    filename = f"Detections/Scenario{n}/image{i}.png"
    # Load file
    sc = im.open(filename).convert('RGB')
    loaded_array_bgr = np.array(sc)
    detects.append(loaded_array_bgr)
total_detections = np.array(detects)

angs = []
dists = []
for i in range(0, len(total_points)):
    angles = []
    distances = []
    for j in range(0, len(total_points[i])):
        if 1.68 * np.pi <= total_points[i][j, 0] <= 2 * np.pi or 0 <= total_points[i][j, 0] <= 0.32 * np.pi:
            if total_points[i][j, 1] <= 20:
                angles.append(total_points[i][j, 0])
                distances.append(total_points[i][j, 1])
    angs.append(angles)
    dists.append(distances)
all_images = copy.deepcopy(total_images)
all_points = copy.deepcopy(total_points)

```

```python

```

```python
def save_history(class_id,x, y, x_lidar, y_lidar):
    if class_id not in vehicle_history:
        vehicle_history[class_id] = {
            "x":[],
            "y":[],
            "xlidar":[],
            "ylidar":[]
        }

    vehicle_history[class_id]["x"].append(x)
    vehicle_history[class_id]["y"].append(y)
    vehicle_history[class_id]["xlidar"].append(x_lidar)
    vehicle_history[class_id]["ylidar"].append(y_lidar)


def wrap_to_pi(angle):
    """Normalize an angle to the interval [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class AngleUnwrapper:
    """
    Keeps yaw continuous for each tracked vehicle ID.
    atan2 gives angles in [-pi, pi], so when the vehicle crosses the boundary
    the value jumps. This class adds/subtracts 2*pi to remove that artificial jump.
    """
    def __init__(self):
        self.prev_wrapped = {}
        self.unwrapped = {}

    def update(self, class_id, yaw_wrapped):
        if class_id not in self.prev_wrapped:
            self.prev_wrapped[class_id] = yaw_wrapped
            self.unwrapped[class_id] = yaw_wrapped
            return yaw_wrapped

        delta = wrap_to_pi(yaw_wrapped - self.prev_wrapped[class_id])
        self.unwrapped[class_id] += delta
        self.prev_wrapped[class_id] = yaw_wrapped
        return self.unwrapped[class_id]

def polar_to_cartesian(ranges, angles):
    ranges = np.array(ranges)
    ranges1 = ranges
    angles = np.array(angles)
    xs = ranges1 * np.cos(angles)
    ys = ranges1 * np.sin(angles)
    zs = np.zeros_like(xs)
    pts = np.stack([xs, ys, zs], axis=1)   # (N,3)
    return pts

def project_points_lidar_to_image(ranges, angles, T_cam_lid, K, image, point_size=5):

    pts_lidar = polar_to_cartesian(ranges, angles)                # (N,3)
    dbscan = DBSCAN(eps=0.35, min_samples=8) #Distance >=8 <10
    labels = dbscan.fit_predict(pts_lidar)
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(0)  # deterministic colors
    colors = {}
    for label in unique_labels:
        if label == -1:
            colors[label] = (180, 180, 180)  # gray for noise
        else:
            colors[label] = tuple(int(c) for c in rng.integers(0, 255, 3))
    N = pts_lidar.shape[0]
    homo = np.hstack([pts_lidar, np.ones((N,1))])                 # (N,4)
    pts_cam_h = (T_cam_lid @ homo.T).T                           # (N,4)
    pts_cam = pts_cam_h[:, :3]
    Xc = pts_cam[:,0]; Yc = pts_cam[:,1]; Zc = pts_cam[:,2]

    x_rot = Yc
    y_rot = -Xc

    # Project to pixels
    u = (fx * -Yc / Zc) + cx
    v = (fy * Xc / Zc) + cy

    # prepare image
    W = int(max(820, cx*2))   # fallback sizes
    H = int(max(410, cy*2))
    if image is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        canvas = image.copy()
        H, W = canvas.shape[:2]

    pixels = []
    for i in range(N):
        #if not valid[i]:
            #continue
        ui = int(round( u[i])); vi = int(round(v[i]))
        #if 0 <= ui < W and 0 <= vi < H:
            # draw small circle (red) for that lidar hit
        color = colors[labels[i]]
        cv2.circle(canvas, (ui, vi), point_size, color, thickness=-1)
        pixels.append((ui, vi, ranges[i], angles[i]))
    return canvas, pixels

def depth_filter_bbox_points(
    pts_box,
    class_id,
    vehicle_history,
    bbox,
    min_points=2,
    max_depth_gap=1.0,
    prev_abs_gate=2.0,
    prev_rel_gate=0.10,
    init_abs_gate=4.0,
    init_rel_gate=0.35
):
    """
    Filter projected LiDAR points inside a YOLO bounding box using depth consistency.

    pts_box columns:
        0 -> image u
        1 -> image v
        2 -> LiDAR range
        3 -> LiDAR angle

    bbox:
        [x1, y1, x2, y2]

    K:
        camera intrinsic matrix
    """

    pts_box = np.asarray(pts_box)

    if len(pts_box) == 0:
        return pts_box

    ranges = pts_box[:, 2].astype(float)

    # ----------------------------------------------------
    # Split points into depth clusters
    # ----------------------------------------------------
    order = np.argsort(ranges)
    pts_sorted = pts_box[order]
    ranges_sorted = ranges[order]

    dynamic_gap = max(max_depth_gap, 0.05 * np.median(ranges_sorted))

    split_indices = np.where(np.diff(ranges_sorted) > dynamic_gap)[0] + 1
    clusters = np.split(pts_sorted, split_indices)

    valid_clusters = [cluster for cluster in clusters if len(cluster) >= min_points]

    if len(valid_clusters) == 0:
        return np.empty((0, pts_box.shape[1]))

    # ----------------------------------------------------
    # Case 1: vehicle already has previous LiDAR distance
    # ----------------------------------------------------
    if class_id in vehicle_history and len(vehicle_history[class_id]["xlidar"]) > 0:
        prev_x = vehicle_history[class_id]["xlidar"][-1]
        prev_y = vehicle_history[class_id]["ylidar"][-1]
        prev_distance = np.sqrt(prev_x**2 + prev_y**2)

        gate = max(prev_abs_gate, prev_rel_gate * prev_distance)

        best_cluster = None
        best_error = np.inf

        for cluster in valid_clusters:
            cluster_distance = np.median(cluster[:, 2])
            error = abs(cluster_distance - prev_distance)

            if error < best_error:
                best_error = error
                best_cluster = cluster

        if best_error <= gate:
            return best_cluster
        else:
            return np.empty((0, pts_box.shape[1]))

    # ----------------------------------------------------
    # Case 2: first time seeing this vehicle ID
    # Use bounding-box size to estimate approximate depth
    # ----------------------------------------------------
    x1, y1, x2, y2 = bbox

    bbox_width = max(y2 - y1, 1.0)

    expected_distance = 1.9*(CAR_HEIGHT * FOCAL_LENGTH) / bbox_width

    init_gate = max(init_abs_gate, init_rel_gate * expected_distance)

    best_cluster = None
    best_error = np.inf

    for cluster in valid_clusters:
        cluster_distance = np.median(cluster[:, 2])
        error = abs(cluster_distance - expected_distance)

        if error < best_error:
            best_error = error
            best_cluster = cluster

    if best_error <= init_gate:
        return best_cluster

    # If no cluster is consistent with bbox-based expected distance,
    # do not initialize this vehicle in this frame.
    return np.empty((0, pts_box.shape[1]))
```

```python
model_detec = YOLO('C:\\Users\\l_daryel\\Downloads\\best.pt')
model_detec.to('cuda')
FOCAL_LENGTH = 312       # example value in pixels
CAR_HEIGHT = 1.5        # assumed target height in meters
out_data2 = []
vehicle_history = {}
vehicle_sequences = {}
times = []
flag = 0
fps = 0.135

# One unwrapper for the raw yaw and one for the filtered yaw.
# They keep a separate state for each tracked vehicle ID.
yaw_raw_unwrapper = AngleUnwrapper()
yaw_filtered_unwrapper = AngleUnwrapper()

'''
front2body = np.array([
    [0.0,  0.0, 1.0,   0.1930],
    [-1.0, 0.0, 0.0,   0.0   ],
    [0.0, -1.0, 0.0,   0.0953],
    [0.0,  0.0, 0.0,   1.0   ]
], dtype=np.float64)

lid2body = np.array([
    [0.0, -1.0, 0.0,  -0.0108],
    [1.0,  0.0, 0.0,  -0.0001],
    [0.0,  0.0, 1.0,   0.1960],
    [0.0,  0.0, 0.0,   1.0   ]
], dtype=np.float64)
'''
front2body = np.array([
    [ 0.0,  0.0, 1.0,  1.930],
    [-1.0,  0.0, 0.0,  0.0000],
    [ 0.0, -1.0, 0.0,  0.953],
    [ 0.0,  0.0, 0.0,  1.0000],
], dtype=np.float64)

lid2body = np.array([
    [0.0, -1.0, 0.0, -0.108],
    [1.0,  0.0, 0.0, -0.001],
    [0.0,  0.0, 1.0,  1.860],
    [0.0,  0.0, 0.0,  1.0000],
], dtype=np.float64)

front2body[2,3] -= 0.0
front2body[1,3] -= 0.0
front2body[0,3] += 0.0
lid2body[0,3] += 0
lid2body[1,3] += 0.0
lid2body[2,3] -= 0.0

# ---------------------------
# Intrinsic matrix from your vector [fx, fy, cx, cy]
# ---------------------------
fx, fy, cx, cy = 318.86, 312.14, 401.34, 210.5
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

T_cam_lid =    lid2body @ np.linalg.inv(front2body)


for i, img in enumerate(all_images):
    begin = time.time()
    image = img.copy()
    results = model_detec.track(image, persist=True)
    times.append(time.time())
    if len(results[0].boxes) != 0:
        current_time = time.time()
        result = results[0]
        angles = []
        distances = []

        for k in range(0, len(total_points[i])):
            if 1.68 * np.pi <= total_points[i][k, 0] <= 2 * np.pi or 0 <= total_points[i][k, 0] <= 0.32 * np.pi:
                if total_points[i][k, 1] <= 20:
                    angles.append(total_points[i][k, 0])
                    distances.append(total_points[i][k, 1])

        result_img, pts_pixels = project_points_lidar_to_image(distances, angles, T_cam_lid, K, image=image)
        pts_pixels = np.array(pts_pixels)
        pts_in_box = []
        for box in result.boxes:
            conf = box.conf[0]
            print(conf)
            if conf >= 0.2:
                class_id = int(box.id[0]) if box.id is not None else 'None'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask = ((pts_pixels[:, 0] >= x1) &
                        (pts_pixels[:, 0] <= x2) &
                        (pts_pixels[:, 1] >= y1) &
                        (pts_pixels[:, 1] <= y2)
                )

                pts_box_all = pts_pixels[mask]
                pts_in_box = depth_filter_bbox_points(
                    pts_box_all,
                    class_id,
                    vehicle_history,
                    bbox=[x1, y1, x2, y2],
                    min_points=2,
                    max_depth_gap=1.0,
                    prev_abs_gate=2.0,
                    prev_rel_gate=0.10,
                    init_abs_gate=4.0,
                    init_rel_gate=0.35
                )

                cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if len(pts_in_box) > 0:
                    centroid = np.mean(pts_in_box, axis=0)  # [u_mean, v_mean]
                    u_c, v_c = centroid[0].astype(int), centroid[1].astype(int)
                    ranges_lid = pts_in_box[:, 2]
                    angles_lid = pts_in_box[:, 3]
                    cv2.circle(result_img, (u_c, v_c), 10, (0, 255, 255), -1)
                    x_lid = ranges_lid * np.cos(angles_lid)
                    y_lid = ranges_lid * np.sin(angles_lid)

                    centroid_x = np.median(x_lid)
                    centroid_y = np.median(y_lid)

                    distance_est = np.sqrt(centroid_x**2 + centroid_y**2)
                    angle_est = np.arctan2(centroid_y, centroid_x)

                    save_history(class_id,u_c,v_c,centroid_x, centroid_y)

                    if len(vehicle_history[class_id]['x']) >= 2:
                        dx = vehicle_history[class_id]['xlidar'][-1] - vehicle_history[class_id]['xlidar'][-2]
                        dy = vehicle_history[class_id]['ylidar'][-1] - vehicle_history[class_id]['ylidar'][-2]
                        dxv = vehicle_history[class_id]['xlidar'][-1] - vehicle_history[class_id]['xlidar'][-2]
                        dyv = vehicle_history[class_id]['ylidar'][-1] - vehicle_history[class_id]['ylidar'][-2]
                        dt = max(times[-1] - times[-2], 1e-6)
                        #print(dt)

                        # atan2 gives the correct direction, but it is wrapped to [-pi, pi].
                        yaw_wrapped = np.arctan2(dx, dy)
                        #yaw_wrapped = np.arctan2(dx, dy)

                        # Continuous yaw removes the artificial jump between +pi and -pi.
                        yaw_unwrapped = yaw_raw_unwrapper.update(class_id, yaw_wrapped)
                        speed = np.sqrt(dxv*dxv + dyv*dyv) / dt

                    else:
                        yaw_wrapped = 0.0
                        yaw_unwrapped = yaw_raw_unwrapper.update(class_id, yaw_wrapped)
                        speed = 0.0

                    # Output columns: distance, relative bearing angle, continuous filtered yaw, filtered speed.
                    out_data2.append([distance_est, angle_est, yaw_unwrapped, speed])

                    cv2.putText(result_img,
                            f'ID:{class_id}, Dist:{distance_est:.1f}m, Angle: {np.degrees(angle_est):.2f}',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)


        # Show frame
        cv2.imshow('Front', result_img)
        #sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

    finish = time.time()
    t = finish - begin
    sleep_time = fps - t
    if sleep_time > 0:
        sleep(sleep_time)
    else:
        print("Sleep time is 0 or negative")
    print(t)


cv2.destroyWindow('Front')

dir=f"C:/Users/l_daryel/Downloads/Conference extension/Data"
os.makedirs(dir, exist_ok=True)
filename = os.path.join(dir, f"sample{n}.txt")
np.savetxt(filename, out_data2, fmt="%.6f")


```

```python
print(out_data2)
```

```python

```