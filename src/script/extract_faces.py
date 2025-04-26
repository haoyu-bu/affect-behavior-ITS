import os, glob
import dlib
from PIL import Image
import numpy as np

print(dlib.DLIB_USE_CUDA)
file_paths = glob.glob('data/raw_images/*.png')

outfolder = 'data/faces/*.png'
print(len(file_paths))
file_paths.reverse()

cnn_face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')

for idx, file_path in enumerate(sorted(file_paths)):
    print(idx, file_path)

    split_list = file_path.split('/')
    basename = split_list[-1]
    video_name = split_list[-2]
    image_name = basename.split('.')[0]
    
    outname = outfolder  + '/' + video_name + '_faces/' + image_name + '.png'
    output_folder = outfolder + '/' + video_name + '_faces/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = Image.open(file_path)
    img = img.convert('RGB')

    img_np = np.array(img)

    dets = cnn_face_detector(img_np, 1)

    if dets:
        det = dets[0]
        # Get x_min, y_min, x_max, y_max, conf
        x_min = det.rect.left()
        y_min = det.rect.top()
        x_max = det.rect.right()
        y_max = det.rect.bottom()
        conf = det.confidence
        if conf > 1.0:
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(img_np.shape[1], x_max); y_max = min(img_np.shape[0], y_max)
            # Crop image
            img_np = img_np[int(y_min):int(y_max),int(x_min):int(x_max)]
            img = Image.fromarray(img_np)
    
    img.save(outname)
