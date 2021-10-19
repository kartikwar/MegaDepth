import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import os
import cv2
import time

IMG_SIZE = 1000
GAUSSIAN_KERNEL_SIZE = 15
BLUR = 3.0
PRESERVE_THRESH = 0.8

def composite_background(foreground, background, mask):
	mask = mask[:,:, np.newaxis]
	background = cv2.resize(background, (mask.shape[1], mask.shape[0]))
	result = mask * foreground.astype(np.float32) + (1 - mask) * background.astype(np.float32)
	# result = result*255.0
	result = np.clip(result, 0, 255).astype(np.uint8)
	# return result, foreground, std_background-std_background_2
	return result

def apply_blur(background, blur_value):
	#done because sending 0 value in opencv does auto blur, which is not intended
	blur_value = max(blur_value, 0.0001)
	background = cv2.GaussianBlur(background,(GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE),sigmaX=blur_value, sigmaY=blur_value)
	background = cv2.GaussianBlur(background,(GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE),sigmaX=blur_value, sigmaY=blur_value)
	# background = cv2.GaussianBlur(background,(GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE),sigmaX=blur_value, sigmaY=blur_value)
	return background

def test_simple(model, img_path):
	total_loss =0 
	toal_count = 0
	print("============================= TEST ============================")
	model.switch_to_eval()

	img = np.float32(io.imread(img_path))/255.0
	img = resize(img, (input_height, input_width), order = 1)
	input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
	input_img = input_img.unsqueeze(0)

	input_images = Variable(input_img.cuda() )
	pred_log_depth = model.netG.forward(input_images) 
	pred_log_depth = torch.squeeze(pred_log_depth)

	pred_depth = torch.exp(pred_log_depth)

	# visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
	# you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
	pred_inv_depth = 1/pred_depth
	pred_inv_depth = pred_inv_depth.data.cpu().numpy()
	# you might also use percentile for better visualization
	pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

	# io.imsave('demo.png', pred_inv_depth)
	# print(pred_inv_depth.shape)
	# sys.exit()
	return pred_inv_depth

def get_file_list(folder_path):
	file_lst = os.listdir(folder_path)
	file_lst = [f_path for f_path in file_lst if ('.jpg' in f_path  or '.png' in f_path)]
	return file_lst

def convert_depth_map_to_color(depth_path):
	depth_im = cv2.imread(depth_path, 0)
	depth_im = (255-depth_im)
	heatmap = cv2.applyColorMap(depth_im, cv2.COLORMAP_JET)
	# cv2.imwrite(depth_path.replace('.png', '_heatmap.png'), heatmap)
	return heatmap

def depth_to_bokeh(im_path, depth_path):
	image = cv2.imread(im_path)
	
	mask = cv2.imread(depth_path, 0)

	height, width, _ = image.shape

	mask = mask/255.0
	# mask[:,:] = 0.5
	thresh = min(np.max(mask), PRESERVE_THRESH)
	mask = np.where(mask >= thresh, mask, mask - (BLUR-1)/10.0 )
	mask = np.clip(mask, 0, 1.0)

	#resize image and depth map to (1000,1000)
	image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
	mask = cv2.resize(mask, (IMG_SIZE,IMG_SIZE))
	
	background = apply_blur(image.copy(), BLUR)

	#to do : composite depth_gaussian with original image based on depth mask
	result = composite_background(image, background, mask)

	return cv2.resize(result,(width, height))

if __name__ == '__main__':
	model = create_model(opt)

	input_height = 384
	input_width  = 512

	start_time = time.time()

	folder_path = '/home/ubuntu/kartik/MegaDepth/val_dataset'

	images_folder = os.path.join(folder_path, 'images')
	depth_folder = os.path.join(folder_path, 'depth')
	heatmap_folder = os.path.join(folder_path, 'heatmap')
	bokeh_folder = os.path.join(folder_path, 'bokeh')
	bg_rem_folder = os.path.join(folder_path, 'bg_remove')
	result_folder = os.path.join(folder_path, 'results')

	if not os.path.exists(depth_folder):
		os.makedirs(depth_folder)
	
	if not os.path.exists(heatmap_folder):
		os.makedirs(heatmap_folder)
	
	if not os.path.exists(bokeh_folder):
		os.makedirs(bokeh_folder)

	if not os.path.exists(result_folder):
		os.makedirs(result_folder)

	file_lst = get_file_list(images_folder)

	for f_path in file_lst:
		im_path = os.path.join(images_folder, f_path)

		bg_path = os.path.join(bg_rem_folder, f_path.replace('.jpg', ".png"))

		try:
			#to do :- remove later
			cv2.imwrite(im_path, cv2.resize(cv2.imread(im_path), (1000,1000)))

			cv2.imwrite(bg_path, cv2.resize(cv2.imread(bg_path, cv2.IMREAD_UNCHANGED), (1000,1000)))

			depth_path = os.path.join(depth_folder, f_path)
			heatmap_path = os.path.join(heatmap_folder, f_path)
			bokeh_path = os.path.join(bokeh_folder, f_path)
			
			result_path = os.path.join(result_folder, f_path)

			depth = test_simple(model, im_path)
			io.imsave(depth_path, depth)
			heatmap = convert_depth_map_to_color(depth_path)
			bokeh = depth_to_bokeh(im_path, depth_path)
			heatmap = cv2.resize(heatmap, bokeh.shape[:2])

			#composite bokeh with bg remove output
			bg_rem_out = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
			mask = bg_rem_out[:,:,3]/255.0
			foreground = bg_rem_out[:,:,:3]
			bokeh_with_bg = composite_background(foreground, bokeh, mask)
			
			cv2.imwrite(heatmap_path, heatmap)
			cv2.imwrite(bokeh_path, bokeh)
			cv2.imwrite(result_path, bokeh_with_bg)
		except Exception as ex:
			continue

	end_time = time.time()
	time_taken = end_time - start_time
	avg_time = (time_taken*1.0)/len(file_lst)

	print("avg time per image is " + str(avg_time))
