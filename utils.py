import os
import numpy as np

def load_image_names(folder, ending=None):
    """ It loads the names of all images from directory. If given an ending,
    only those files will be returned.	
    """
    img_files = sorted(os.listdir(folder)) 
    if ending is not None:
      img_files = [ii for ii in img_files if ii.endswith(ending)]
    img_number = len(img_files)     
    print("  Loading names of", str(img_number), "images...")   
    return np.asarray(img_files)