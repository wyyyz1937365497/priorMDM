import numpy as np
import json

# 加载.npy文件
data = np.load("results.npy", allow_pickle=True).item()
output = {
    "motions": data['motion'].tolist(),
    "texts": data['text'],
    "edges": [
        (18, 20), (16, 18), (13, 16), (12, 15), (14, 17), (17, 19), 
        (19, 21), (9, 6), (6, 3), (3, 0), (0, 1), (0, 2), 
        (1, 4), (2, 5), (4, 7), (7, 10), (5, 8), (8, 11)
    ],
    "bone_names": [
        "Hand_L", "Arm_L", "Shoulder_L", "Head", "Shoulder_R", "Arm_R", 
        "Hand_R", "Chest", "Spine", "Hips", "Hips_L", "Hips_R", 
        "Thigh_L", "Thigh_R", "Leg_L", "Foot_L", "Leg_R", "Foot_R"
    ]
}

# 保存为JSON
with open("motion_data.json", "w") as f:
    json.dump(output, f)