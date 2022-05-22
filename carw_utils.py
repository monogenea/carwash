'''
GENERAL UTILITIES FOR THE CARWASH PROJECT
'''
import os
import re
import cv2
import numpy as numpy
import pandas as pd

def setup_wdir(pars_url: str) -> None:
    """Project env / working directory setup
    
    Args:
        pars_url (str): URL to a valid PaddleSeg model parameter config (.pdparams).
    
    Returns:
        None
    
    """
    # git clone if paddleseg not available
    if not os.path.exists('PaddleSeg/'):
        print('Cloning PaddleSeg project...')
        os.system('git clone https://github.com/PaddlePaddle/PaddleSeg')
    # download and export model if not available
    if not os.path.exists('PaddleSeg/carw_config'):
        print('Downloading and exporting model...')
        # extract configuration YAML and base dir
        cfg_yml = pars_url.split('/')[-2]
        cfg_dir = cfg_yml.split('_')[0]
        os.system(f'cd PaddleSeg && mkdir {cfg_dir} && cd {cfg_dir} && wget {pars_url}')
        # export model
        cmd = f'''
            python3 PaddleSeg/export.py \
                --config PaddleSeg/configs/{cfg_dir}/{cfg_yml}.yml \
                --model_path PaddleSeg/{cfg_dir}/model.pdparams \
                --save_dir PaddleSeg/carw_config \
                --input_shape 1 3 1024 1024
            '''
        os.system(cmd)
    print('Environment successfully set up!')

    return None

def run_inference(img_path: str) -> None:
    """Run image through inference with a specified PaddleSeg configuration,
    write resulting segmented image into a results/ directory
    
    Args:
        img_path (str): Path to target image (JPG or PNG).

    Returns:
        None
    
    """
    cmd = f'''
          python3 PaddleSeg/deploy/python/infer.py \
              --config PaddleSeg/carw_config/deploy.yaml \
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

def score_dirt(fpath: str, cdir: str) -> numpy.ndarray:
    """End-to-end process to run PaddleSeg inference and score dirt level
    
    Args:
        fpath (str): Path to target image (JPG or PNG).
        cdir (str): Directory name of PaddleSeg configuration.
    
    Returns:
        float: Dirt score for tested car (0-1 range).
    
    """
    # get path to mask object
    fpath_mask = fpath.replace('data/', 'output/').replace('.jpg', '.png')
    # inference
    run_inference(img_path=fpath, cfg_dir=cdir)
    # segment 
    seg_car = segment_car(img_path=fpath, mask_path=fpath_mask)
    # Canny edge detection (binarized)
    edges = cv2.Canny(image=seg_car, threshold1=100, threshold2=200) / 255
    # init kernel, apply morph closing (2x)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # compute score
    score = np.mean(closing[seg_car[..., 0] != 0])

    return score