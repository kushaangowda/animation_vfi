import numpy as np
from PIL import Image
from skimage import color
import h5py
import os
from tqdm import tqdm

def process_images(image_folder, flows_folder, total_images=None):
    count = 0
    with h5py.File('../image_data.h5', 'w') as hf:
        file_paths = os.listdir(image_folder)
        
        if total_images is None:
            total_images = len(file_paths)
            
        for image_filename in tqdm(file_paths):
            if count == total_images:
                break
            try:
                frame1_path = os.path.join(image_folder, image_filename, "frame1.png")
                frame2_path = os.path.join(image_folder, image_filename, "frame2.png")
                frame3_path = os.path.join(image_folder, image_filename, "frame3.png")
                
                frame1 = np.array(Image.open(frame1_path), dtype=np.uint8)
                frame2 = np.array(Image.open(frame2_path), dtype=np.uint8)
                frame3 = np.array(Image.open(frame3_path), dtype=np.uint8)
                
                flow13_path = os.path.join(flows_folder, image_filename, "guide_flo13.npy")
                flow31_path = os.path.join(flows_folder, image_filename, "guide_flo31.npy")
                
                flow13 = np.load(flow13_path).astype(np.float16)
                flow31 = np.load(flow31_path).astype(np.float16)
                
                group = hf.create_group(image_filename)
                group.create_dataset('frame1', data=frame1)
                group.create_dataset('frame2', data=frame2)
                group.create_dataset('frame3', data=frame3)
                group.create_dataset('flow13', data=flow13)
                group.create_dataset('flow31', data=flow31)

                count += 1
                
            except Exception as e:
                print(e)
                raise Exception("")


if __name__ == '__main__':

    process_images(
        '../datasets/test_2k_540p/', 
        '../datasets/test_2k_pre_calc_sgm_flows/',
        total_images=None
    )