import numpy as np
import mathutils as mu

import bmesh
import bpy

# 加载.npy文件
ndir = "G://AI//priorMDM//wyyyz//results.npy"
data = np.load(ndir, allow_pickle=True)

# 获取motion数据
motions = data[()]['motion']
print("Frames:", motions.shape[3])  # 打印帧数
print("Joints:", motions.shape[1])  # 打印关节数
print("Motions:", len(motions))  # 打印动作数
assert motions.shape[2] == 3, "Could not find XYZ data"  # 确认数据维度为3

# 获取动作索引
m_index = 0
print("Prompt:", data[()]['text'][m_index])  # 打印动作描述
joints = []
# 遍历动作中的每个关节
for i in motions[m_index]:
    vectors = []
    # 遍历每个关节的位置
    for pos in range(i.shape[1]):
        # Z > X, X > Y, Y > Z
        vectors.append(mu.Vector([i[2, pos], i[0, pos], i[1, pos]]))        
    joints.append(vectors)
    
assert len(joints) == 22  # 确认关节数

# 获取关节位置和边
verts = [i[0] for i in joints]
edges = [
    (18, 20), (16, 18), (13, 16), (12, 15), (14, 17), (17, 19), 
    (19, 21), (9, 6), (6, 3), (3, 0), (0, 1), (0, 2), 
    (1, 4), (2, 5), (4, 7), (7, 10), (5, 8), (8, 11),
]

# 获取关节名称
bone_names = [
    "Hand_L", "Arm_L", "Shoulder_L", "Head", "Shoulder_R", "Arm_R", 
    "Hand_R", "Chest", "Spine", "Hips", "Hips_L", "Hips_R", 
    "Thigh_L", "Thigh_R", "Leg_L", "Foot_L", "Leg_R", "Foot_R", 
]

# 创建网格和对象
mesh = bpy.data.meshes.new("bpy_mesh")
obj = bpy.data.objects.new("bpy_object", mesh)
col = bpy.data.collections["Collection"]
col.objects.link(obj)
bpy.context.view_layer.objects.active = obj
#averts = []
#for v in joints:
#    averts.extend(v)
#mesh.from_pydata(averts, [], [])

# TODO: Is this really the best way to create armature in Blender?
# 创建骨架
bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', scale=(1, 1, 1))
print("Armatures:", len(bpy.data.armatures))
ature = bpy.data.armatures[-1]
ature_name = ature.name

# 进入编辑模式
bpy.ops.object.mode_set(mode='EDIT', toggle=False)
ebon = ature.edit_bones
# Clear all existing
for bone in ebon:
    ebon.remove(bone)

for ie, e in enumerate(edges):
    b = ebon.new(bone_names[ie])
    b.head = joints[e[0]][0]
    b.tail = joints[e[1]][0]

bpy.ops.object.mode_set(mode='OBJECT')
ature = bpy.data.objects["骨架"]
apose = ature.pose

# Blender uses Y-axis ref for bone angle calc from head and tail
# Add animation frames
for f in range(len(joints[0])):
    for ie, e in enumerate(edges):
        b = apose.bones[bone_names[ie]]

        jh = joints[e[0]][f]
        jt = joints[e[1]][f]
        jdif = (jt - jh).normalized()
        rto = mu.Vector([0,1,0]).rotation_difference(jdif)

        mp = ature.convert_space(pose_bone=b,
            matrix=b.matrix, from_space='POSE', to_space='WORLD')
            
        mat_loc = mu.Matrix.Translation(jh)
        mp = mat_loc @ rto.to_matrix().to_4x4()
        
        b.matrix = ature.convert_space(pose_bone=b,
            matrix=mp, from_space='WORLD', to_space='POSE')

        b.keyframe_insert("rotation_quaternion", frame=f)
        b.keyframe_insert("location", frame=f)