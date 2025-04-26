import pandas as pd
import numpy as np
import glob, os
import sklearn
from sklearn.preprocessing import StandardScaler

raw_paths = glob.glob('data/openface/*.csv')

lengths = []
for idx, path in enumerate(raw_paths):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '')
    
    # OpenFace features
    if len(df) < 20:
        print(path)
    df = df.iloc[::10, :] # downsample
    lengths.append(len(df))
    gaze = df[['gaze_angle_x', 'gaze_angle_y', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']]
    pose = df[['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
    au_intensity = df[['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']]
    au_presence = df[['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']]
    gaze = np.array(gaze)
    pose = np.array(pose)
    au_intensity = np.array(au_intensity)
    au_presence = np.array(au_presence)
    features = np.concatenate([gaze, pose, au_intensity, au_presence], axis=1)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    vidnum = path.split('/')[-1].split('.')[0]
    dir_path = 'data/openface-processed-feats/'
    if not os.path.exists(dir_path):
       os.mkdir(dir_path)
    savepath = os.path.join(dir_path, vidnum + '_ts_scaled_sparse.npy')
    print(savepath, features.shape)
    np.save(savepath, features)

print(lengths)
print(max(lengths))
print(np.average(lengths))
