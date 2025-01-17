from model import get_unet
from utils import load_image_names
import argparse
import imageio
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import morphology, measure, io
from skimage.segmentation import clear_border


data_folder = './'
image_shape = (512, 512, 1)   
data_format = 'channels_last' 

# The network
classes = (0, 1)
growth_rate = 8
blocks = (6, 12, 24, 48, 24, 12, 6)
learning_rate = 0.001
learning_rate_decay = 0.99  

# Folder to the weights:
edge_weights = data_folder + 'weights/weights_edges_epoch150.h5'
roi_weights  = data_folder + 'weights/weights_ROI_epoch150.h5'


print("Loading the model...")
##############################################################################
# Load the Dense U-net model for CNN-Edge
model_edge = get_unet(image_shape, 
                      blocks, 
                      growth_rate=growth_rate, 
                      learning_rate=learning_rate,
                      weights=edge_weights,
                      data_format=data_format)  
print("Model loaded successfully.")

# Number of parameters in each model: 7.1M

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_folder', type=str, default='./data/new_crop')
    parser.add_argument('-o','--output_folder', type=str, default='./data/new_crop_edge')
    parser.add_argument('-t','--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    intn_names = load_image_names(input_folder, ending='png')
    intn_image = np.zeros((len(intn_names), image_shape[0], image_shape[1], 1), dtype=np.float32)
    
    for ii in range(len(intn_names)):
        a_name = os.path.join(input_folder, intn_names[ii])
        img = imageio.imread(a_name).astype(np.uint16)
        # normalize to 0-255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        intn_image[ii] = img[:,:,np.newaxis]  # 添加一个新的维度
    
    intn_image = intn_image.astype(np.float32) / 255     # Normalize

    # predict the edge image
    print("Predicting the edge image...")
    edge_image = model_edge.predict(intn_image)
    print(edge_image.shape)
    edge_image = edge_image[:, :, 1]
    edge_image = edge_image.reshape(-1, image_shape[0], image_shape[1])
    print(edge_image.shape)

    # if the output folder does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # save the edge image
    # print("Saving the edge image...")
    # for ii in range(len(intn_names)):
    #     edge_name = os.path.join(output_folder, intn_names[ii].replace('.png', '_edge.png'))
    #     imageio.imwrite(edge_name, (edge_image[ii] * 255).astype(np.uint8))
    # print("Edge image saved successfully.")

    # threshold the edge image
    threshold = args.threshold
    binary_images = edge_image > threshold
    print(binary_images.shape)

    for ii, binary_image in enumerate(binary_images):
        # 定义结构元素
        selem = morphology.disk(4)
        # 闭运算连接
        binary_image = morphology.closing(binary_image, selem)
        # binary = morphology.remove_small_objects(binary, min_size=100)

        # 仅爆留骨架
        binary_image = morphology.skeletonize(binary_image)

        # 膨胀
        binary_image = morphology.dilation(binary_image, morphology.square(2))

        binary_name = os.path.join(output_folder, intn_names[ii].replace('.png', '_edge_binary.png'))
        imageio.imwrite(binary_name, (binary_image * 255).astype(np.uint8))

    
        binary_image = ~binary_image
        
        binary_no_border = clear_border(binary_image)

        # 连通域标记
        label_image = measure.label(binary_no_border, connectivity=2)
        
        # 将非0区域统一标记为1
        label_image[label_image > 0] = 1
    
        # 保存连通域标记图像
        label_name = os.path.join(output_folder, intn_names[ii].replace('.png', '_cell_label.png'))
        imageio.imwrite(label_name, (label_image * 255).astype(np.uint8))

