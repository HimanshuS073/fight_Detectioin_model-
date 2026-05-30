import os
import shutil

p = "/home/himanshus/Projects/fight_detection_model/normal_video"
v = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
o = os.path.join(p, "all_videos")
os.makedirs(o, exist_ok=True)

k = 0

for i in os.listdir(p):
    d = os.path.join(p, i)
    if not os.path.isdir(d):
        continue
    for j in os.listdir(d):
        s = os.path.join(d, j)
        if os.path.isfile(s):
            _, e = os.path.splitext(j)
            if e.lower() in v:
                n = os.path.join(o, str(k) + e.lower())
                shutil.copy2(s, n)
                k += 1
