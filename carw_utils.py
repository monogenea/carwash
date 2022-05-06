'''
GENERAL UTILITIES FOR THE CARWASH PROJECT
'''
import os
import re
import cv2
import numpy as numpy
import pandas as pd

# we assume the file is a tar.gz compressed object
FILE_PTTN = re.compile('[^/]+\.tar\.gz')

def setup_wdir(cfg_url: str) -> None:
    """Project env / working directory setup
    
    Args:
        cfg_url (str): URL to a valid PaddleSeg configuration tar.gz file.

    Returns:
        None
    
    """
    # determine name of config file
    fname = re.findall(pattern=FILE_PTTN, string=cfg_url)[0]
    # git clone if paddleseg not available
    if not os.path.exists('PaddleSeg/'):
        print('Cloning PaddleSeg project...')
        os.system('git clone https://github.com/PaddlePaddle/PaddleSeg')
    # download and extract model config if not available
    if not os.path.exists(f'PaddleSeg/{fname.replace(".tar.gz", "")}/'):
        print('Downloading model config...')
        os.system(f'cd PaddleSeg && wget {cfg_url} && tar -zxvf {fname} && rm {fname}')
    print('Env successfully set up')

    return None

def run_inference(img_path: str, cfg_dir: str) -> None:
    """Run image through inference with a specified PaddleSeg configuration,
    write resulting segmented image into a results/ directory
    
    Args:
        img_path (str): Path to target image (JPG or PNG).
        cfg_dir (str): Directory name of PaddleSeg configuration.

    Returns:
        None
    
    """
    cmd = f'''
          python3 PaddleSeg/deploy/python/infer.py \
              --config PaddleSeg/{cfg_dir}/deploy.yaml \
              --image_path {img_path} \
              --save_dir output/
          '''
    os.system(cmd)

    return None

def segment_car(img_path: str, mask_path: str) -> numpy.ndarray:
    """Load PaddleSeg prediction and extract car component(s)
    
    Args:
        img_path (str): Path to target image (JPG or PNG).
        mask_path (str): Path to predicted segmentation of target image (PNG).
    
    Returns:
        numpy.ndarray: Zeroed RGB image carrying the car component(s)
    
    """
    # load original image and segmentation mask
    img = cv2.imread(img_path, 1)
    mask = cv2.imread(mask_path, 1)
    # extract car mask
    car_mask = cv2.inRange(mask, (128, 128, 64), (128, 128, 64))
    # segment car component(s)
    out = cv2.bitwise_and(img, cv2.merge((car_mask, car_mask, car_mask)))
    
    return out

def get_car(fpath: str, cdir: str) -> numpy.ndarray:
    """End-to-end process to run PaddleSeg inference and extract car component(s)
    
    Args:
        fpath (str): Path to target image (JPG or PNG).
        cdir (str): Directory name of PaddleSeg configuration.
    
    Returns:
        numpy.ndarray: Zeroed RGB image carrying the car component(s)
    
    """
    # get path to mask object
    fpath_mask = fpath.replace('data/', 'output/').replace('.jpg', '.png')
    # inference
    run_inference(img_path=fpath, cfg_dir=cdir)
    # segment 
    out = segment_car(img_path=fpath, mask_path=fpath_mask)

    return out