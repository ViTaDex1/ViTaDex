import os
import copy
import glob
import yaml
import cv2 as cv
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.batch import Data as GraphData

from model import TactileGCN,MergeModel,ImageCNN
from pose_loss import Pose6DLoss
from dataset import VisionDataset,MergeDataset,TactileDataset, VisionDataset_t, TactileDataset_t, MergeDataset_t


Abs_Path=os.path.dirname(os.path.abspath(__file__))

torch.set_printoptions(
    precision=2,    
    threshold=1000,
    edgeitems=3,
    linewidth=150,  
    profile=None,
    sci_mode=False  
)

np.set_printoptions(precision=3,suppress=True)

###################################Basic function###################################
def matrix_from_quaternion(quaternion, pos=None):
    """
    Return homogeneous rotation matrix from quaternion.
    Input is qx qy qz qw
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    q = q[[3, 0, 1, 2]]   # to w x y z

    if pos is None: pos = np.zeros(3)

    n = np.dot(q, q)
    if n < 1e-20:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], pos[0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], pos[1]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], pos[2]],
        [0.0, 0.0, 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
	"""Return quaternion from rotation matrix.

	LAYOUT : [X, Y, Z, W]

	If isprecise is True, the input matrix is assumed to be a precise rotation
	matrix and a faster algorithm is used.
	"""
	M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
	if isprecise:
		q = np.empty((4,))
		t = np.trace(M)
		if t > M[3, 3]:
			q[0] = t
			q[3] = M[1, 0] - M[0, 1]
			q[2] = M[0, 2] - M[2, 0]
			q[1] = M[2, 1] - M[1, 2]
		else:
			i, j, k = 1, 2, 3
			if M[1, 1] > M[0, 0]:
				i, j, k = 2, 3, 1
			if M[2, 2] > M[i, i]:
				i, j, k = 3, 1, 2
			t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
			q[i] = t
			q[j] = M[i, j] + M[j, i]
			q[k] = M[k, i] + M[i, k]
			q[3] = M[k, j] - M[j, k]
		q *= 0.5 / np.sqrt(t * M[3, 3])
	else:
		m00 = M[0, 0]
		m01 = M[0, 1]
		m02 = M[0, 2]
		m10 = M[1, 0]
		m11 = M[1, 1]
		m12 = M[1, 2]
		m20 = M[2, 0]
		m21 = M[2, 1]
		m22 = M[2, 2]
		# symmetric matrix K
		K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
					  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
					  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
					  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
		K /= 3.0
		# quaternion is eigenvector of K that corresponds to largest eigenvalue
		w, V = np.linalg.eigh(K)
		q = V[:, np.argmax(w)]
		if q[0] < 0.0:
			np.negative(q, q)
	return q

def project_points(points, K):
    us = np.divide(points[:, 0]*K[0, 0], points[:, 2]) + K[0, 2]
    vs = np.divide(points[:, 1]*K[1, 1], points[:, 2]) + K[1, 2]
    us = np.round(us).astype(np.int32).reshape(-1, 1)
    vs = np.round(vs).astype(np.int32).reshape(-1, 1)
    return np.hstack((us, vs))

def check_noralize(quaternions):
    #检查四元数是否正确归一化
    temp = np.array([quaternions[0], quaternions[1], quaternions[2], quaternions[3]])
    if abs(np.sum(np.square(temp)) -1) < 0.01:
        print("***correct is: {}***".format(np.sum(np.square(temp)) -1))
        return True
    print("sum of quaterion is:{}".format(np.sum(np.square(temp))))
    return False

def tactile_np2tensor(tip_data_array,tip_pose_array):
    tensor_tip_data_array=torch.from_numpy(tip_data_array[np.newaxis,:]).cuda()
    tensor_tip_pose_array=torch.from_numpy(tip_pose_array[np.newaxis,:]).cuda()

    return [tensor_tip_pose_array,tensor_tip_data_array]

def tensor2np(tensor):
    return tensor.detach().cpu().numpy()

def pose_matrix2pose_vector(pose_matrix):
    quaternion=quaternion_from_matrix(pose_matrix)
    xyz=pose_matrix[:3,3]
    pose_vector=np.concatenate([xyz,quaternion])
    return pose_vector

def get_pose_from_yaml(yaml_path):
    """
    The yaml file should be like:

    pose:
    position:
        x: -0.4114404581059961
        y: 0.03589671794642055
        z: 0.5614216934472097
    orientation:
        x: -0.6282095478388904
        y: 0.007425104192635422
        z: 0.03591101479832567
        w: 0.7771795357881858
    """
    data_dict=yaml.load(open(yaml_path))['pose']
    pos=np.array([data_dict['position']['x'],data_dict['position']['y'],data_dict['position']['z']])
    ori=np.array([data_dict['orientation']['x'],data_dict['orientation']['y'],data_dict['orientation']['z'],data_dict['orientation']['w']])
    return np.concatenate([pos,ori],axis=0)

def compute_bbox(pose, K, scale_size=300, scale=(1, 1, 1)):
    obj_x = pose[0, 3] * scale[0]
    obj_y = pose[1, 3] * scale[1]
    obj_z = pose[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float32)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
    points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
    points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
    points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
    projected_vus = np.zeros((points.shape[0], 2))
    projected_vus[:, 1] = points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]
    projected_vus[:, 0] = points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]
    projected_vus = np.round(projected_vus).astype(np.int32)
    return projected_vus

def crop_bbox(color, depth, boundingbox, output_size=(128, 128), seg=None):
    left = np.min(boundingbox[:, 1])
    right = np.max(boundingbox[:, 1])
    top = np.min(boundingbox[:, 0])
    bottom = np.max(boundingbox[:, 0])

    h, w, c = color.shape
    crop_w = right - left
    crop_h = bottom - top
    color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
    depth_crop = np.zeros((crop_h, crop_w), dtype=np.float32)
    seg_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
    top_offset = abs(min(top, 0))
    bottom_offset = min(crop_h - (bottom - h), crop_h)
    right_offset = min(crop_w - (right - w), crop_w)
    left_offset = abs(min(left, 0))

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, h)
    right = min(right, w)
    color_crop[top_offset:bottom_offset, left_offset:right_offset,
               :] = color[top:bottom, left:right, :]
    depth_crop[top_offset:bottom_offset,
               left_offset:right_offset] = depth[top:bottom, left:right]
    resized_rgb = cv.resize(color_crop, output_size,
                             interpolation=cv.INTER_NEAREST)
    resized_depth = cv.resize(
        depth_crop, output_size, interpolation=cv.INTER_NEAREST)

    if seg is not None:
        seg_crop[top_offset:bottom_offset,
                 left_offset:right_offset] = seg[top:bottom, left:right]
        resized_seg = cv.resize(
            seg_crop, output_size, interpolation=cv.INTER_NEAREST)
        final_seg = resized_seg.copy()

    mask_rgb = resized_rgb != 0
    mask_depth = resized_depth != 0
    resized_depth = resized_depth.astype(np.uint16)
    final_rgb = resized_rgb * mask_rgb
    final_depth = resized_depth * mask_depth
    if seg is not None:
        return final_rgb, final_depth, final_seg
    else:
        return final_rgb, final_depth

def get_colormap_from_depth(depth):
    cv_image=depth.copy()
    # ROI=cv2.inRange(cv_image, 0, 1500)
    # cv_image=cv_image*ROI/255
    cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
    cv_image=cv_image.astype(np.uint8)
    color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
    return color_map

def pose_matrix2pose_vector(pose_matrix):
    quaternion=quaternion_from_matrix(pose_matrix)
    xyz=pose_matrix[:3,3]
    pose_vector=np.concatenate([xyz,quaternion])
    return pose_vector

def clean_dataset():
    """
    To see different serial
    0-349
    350-864
    865-2107
    2108-2266
    2267-4457
    """
    dataset_path="/home/Project/Code/code/workspace/src/seedata/scripts/data/WholeDataset/BigDataset"
    all_index_data=glob.glob(dataset_path+"/color*")
    # print(all_index_data)
    
    all_count=0
    while all_count<len(all_index_data):
        print("Now is index {} data".format(all_count))
        image=cv.imread(os.path.join(dataset_path,"color_{}.png".format(all_count)))
        cv.imshow("image",image)
        temp_input=cv.waitKey(0)
        if temp_input==ord('b'):
            all_count=all_count-5
        all_count=all_count+1




####################################Calculate error metric####################################
def normal_quaternion(quaternion):
    return quaternion/(np.sqrt(np.sum(np.square(quaternion))))

def calculate_diff(pose1,pose2):
    """
    Reference metric in "VisuoTactile 6D Pose Estimation of an In-Hand Object Using Vision and Tactile Sensor Data"
    #return in m,rad
    """
    xyz_diff=np.linalg.norm(pose1[:3]-pose2[:3]) * 100.
    quaternion_diff=np.arccos(2*np.square(np.dot(normal_quaternion(pose1[3:]),normal_quaternion(pose2[3:])))-1)
    
    return xyz_diff,quaternion_diff


###################################See data Distribution###################################
def see_error_rate_result():
    pass


###################################To test network generalization###################################
def test_tactile_data():
    object_name="Screwdriver"
    test_path="/media/lab404/dataset2/pose-estimation/VTDexM/Screwdriver"



    network=TactileGCN().cuda()
    network.load_state_dict(torch.load("/media/lab404/dataset2/pose-estimation/ObjectInHand-Dataset/SelectLSTM/log/Baseline-experiments-train/T/models/best_model_epoch490.pth"))
    network.eval()
        
    all_predict_result_list=[]
    all_diff_result_list=[]
    test_dataset = TactileDataset_t(test_path, split_method='shuffle')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    output_dir = os.path.join(Abs_Path, "tactile_predict_results")
    os.makedirs(output_dir, exist_ok=True)
    for index,data in enumerate(test_loader):
        tactile_data, target_pose = data
        target_pose = target_pose.cuda()
        tactile_data = [
            arr.cuda() if torch.is_tensor(arr)
            else torch.from_numpy(arr).float().cuda()
            for arr in tactile_data
        ]

        #network infer
        predict_result=network(tactile_data)

        #get all results
        np_predict_result=predict_result.cpu().detach().numpy().squeeze()
        np_target_pose=target_pose.cpu().detach().numpy().squeeze()
        np.savetxt(
            os.path.join(output_dir, f"predict_{object_name}_{index:04d}.txt"),  # 0000~9999编号
            np_predict_result,  # 保持4x4矩阵格式
            fmt="%.6f",  # 保留6位小数
            delimiter="\t"
        )
        all_predict_result_list.append(np_predict_result)
        all_diff_result_list.append(calculate_diff(np_predict_result,np_target_pose))

    all_predict_array=np.array(all_predict_result_list)
    all_diff_result=np.array(all_diff_result_list)
    
    # train_dataset=TactileDataset(dataset_path,split_method='shuffle')
    # train_index=train_dataset.index_list
    test_dataset=TactileDataset_t(test_path,split_method='shuffle')
    test_index=test_dataset.index_list

    print("***********Object:all results are:***********")
    print("All average result are:")
    np.set_printoptions(precision=8)

    print(np.mean(all_diff_result, axis=0))
    
    
    # np.savez(os.path.join(Abs_Path,"tactile_result.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)

def test_vision_data():
    #1: Init the network

    test_path = "/media/lab404/dataset2/pose-estimation/VTDexM/Tape1"
    object_name = 'Tape1'

    network=ImageCNN().cuda()
    network.load_state_dict(torch.load("/media/lab404/dataset2/pose-estimation/ObjectInHand-Dataset/SelectLSTM/log/Baseline-experiments-train/V/models/best_model_epoch480.pth"))
    network.eval()

    all_predict_result_list=[]
    all_diff_result_list=[]
    
    #2: load index data
    # all_dataset=VisionDataset(dataset_path,split_method='index',split_index_list=[0,10000])
    # all_loader=DataLoader(all_dataset,batch_size=1,shuffle=False)
    test_dataset=VisionDataset_t(test_path,split_method='shuffle')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    output_dir = os.path.join(Abs_Path, "vision_predict_results")
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx,data in enumerate(test_loader):
        rgbd_data,target_pose=data
        rgbd_data,target_pose=rgbd_data.cuda(),target_pose.cuda()
        predict_result=network(rgbd_data.float())
        

        #get all results
        np_predict_result=predict_result.cpu().detach().numpy().squeeze()
        np_target_pose=target_pose.cpu().detach().numpy().squeeze()
        np.savetxt(
            os.path.join(output_dir, f"{object_name}_{batch_idx:04d}.txt"),  # 0000~9999编号
            np_predict_result,  # 保持4x4矩阵格式
            fmt="%.6f",  # 保留6位小数
            delimiter="\t"
        )
        all_predict_result_list.append(np_predict_result)
        all_diff_result_list.append(calculate_diff(np_predict_result,np_target_pose))

    all_diff_result=np.array(all_diff_result_list)

    # train_dataset=VisionDataset(dataset_path,split_method='shuffle')
    # train_index=train_dataset.index_list
    test_dataset=VisionDataset_t(test_path,split_method='shuffle')
    test_index=test_dataset.index_list
    
    print("***********Object:all results are:***********")
    print("All average result are:")
    np.set_printoptions(precision=8)
    print(np.mean(all_diff_result,axis=0))
    
    # print("train average result are:")
    # train_result=all_diff_result[train_index]
    # print(np.mean(train_result,axis=0))
    
    # print("test average result are:")
    # test_result=all_diff_result[test_index]
    # print(test_result.shape)
    # print(np.mean(test_result,axis=0))
    
    
    # np.savez(os.path.join(Abs_Path,"vision_result.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)

def test_merge_data():
    """
    To test different serial generalization
    """
    #1: Init all data

    test_path="/media/lab404/dataset2/pose-estimation/VTDexM/Plastic_water_cup1"
    object_name='Plastic_water_cup1'

    network=MergeModel().cuda()
    network.load_state_dict(torch.load("/media/lab404/dataset2/pose-estimation/ObjectInHand-Dataset/SelectLSTM/log/Baseline-experiments-train/Vision-Tactile--05-27_22-03/models/best_model_epoch600.pth"))
    network.eval()

    test_dataset = MergeDataset_t(test_path, split_method='shuffle')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_diff_result_list=[]
    all_predict_result_list=[]
    output_dir = os.path.join(Abs_Path, "robot_exper")
    os.makedirs(output_dir, exist_ok=True)
        
    for batch_idx,data in enumerate(test_loader):
        tactile_data,rgbd_data,target_pose=data
        tactile_data = [
            arr.cuda() if torch.is_tensor(arr)
            else torch.from_numpy(arr).float().cuda()
            for arr in tactile_data
        ]

        rgbd_data,target_pose=rgbd_data.cuda(),target_pose.cuda()
        predict_result=network(rgbd_data, tactile_data)
        np_predict_result=predict_result.squeeze().detach().cpu().numpy()
        np_target_pose=target_pose.squeeze().detach().cpu().numpy()
        np.savetxt(
            os.path.join(output_dir, f"{object_name}_{batch_idx:04d}.txt"),  # 0000~9999编号
            np_predict_result,  # 保持4x4矩阵格式
            fmt="%.6f",  # 保留6位小数
            delimiter="\t"
        )
        
        
        #error data is:
        diff_result=calculate_diff(np_predict_result,np_target_pose)
        all_diff_result_list.append(diff_result)
        all_predict_result_list.append(np_predict_result)
        
    all_predict_array=np.array(all_predict_result_list)
    all_diff_result=np.array(all_diff_result_list)
        
    # np.savez(os.path.join(Abs_Path,"merge.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)
    
    
    test_dataset=MergeDataset_t(test_path,split_method='shuffle')
    test_index=test_dataset.index_list
    
    print("All result array shape is:")
    print(all_diff_result.shape)
    
    all_diff_result=all_diff_result
    print("All average result are:")
    np.set_printoptions(precision=8)
    print(np.mean(all_diff_result,axis=0))
    

if __name__ == "__main__":
    # generate_select_data()
    
    # test_tactile_data()
    # test_vision_data()
    test_merge_data()
    # test_select_LSTM()
    
    # get_analysis_data()


