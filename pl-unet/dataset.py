import torch
import torch.utils.data as torch_data
import glob
import os
import cv2
import random
from PIL import Image
import numpy as np


class RemoteSenseDataset(torch_data.Dataset):

    def __init__(self, img_dir, transform = None, random_rect_value = 40, block_size = 256, img_suffix = "_img.png", mask_suffix = "_mask.png", enable_cache = False, test_mode = False):
        """
        遥感数据集 torch Dataset
 
        :param img_dir: 存放数据的文件夹，里面包含后缀为img_suffix的原图像和后缀为mask_suffix标注图像
        :param transform: 可调用对象，将dtype=np.uint8的img对象做一些变化
        :param random_rect_value: 切割子图中心点偏移的随机范围大小
        :param block_size: 每个子图大小
        :param img_suffix: 数据集中图像路径后缀
        :param mask_suffix: 数据集中标注路径后缀
        :param enable_cache: 是否将打开后的image和mask进行缓存
        :raises RuntimeError: 如果没有找到符合要求的图像会报错

        """
        img_path_list = glob.glob(os.path.join(img_dir, '*' + img_suffix))
        
        def gen_label_path(img_path):
            return img_path[:-len(img_suffix)] + mask_suffix

        if 0 == len(img_path_list):
            raise RuntimeError("can not find any image in %s" % img_dir)
        
        self.data_list = list()
        self.cache = dict() if enable_cache else None
        self.block_size = block_size
        self.transform = transform
        self.random_rect_value = random_rect_value
        self.test_mode = test_mode

        for img_path in img_path_list:

            img_obj = Image.open(img_path)
            img_width, img_height = img_obj.size

            start_axis = block_size // 2
            width_axes = np.arange(start_axis, img_width, block_size)
            height_axes = np.arange(start_axis, img_height, block_size)
            
            xs, ys = np.meshgrid(width_axes, height_axes)
            center_axis = list(zip(xs.flat, ys.flat))

            for x, y in center_axis:
                self.data_list.append({
                    'img_path': img_path,
                    'mask_path': gen_label_path(img_path),
                    'img_height': img_height,
                    'img_width': img_width,
                    'x': x,
                    'y': y,
                })

    
    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, index):

        img = None
        mask = None
        info_dict = self.data_list[index]
        img_filepath = info_dict['img_path']
        mask_filepath = info_dict['mask_path']
        if isinstance(self.cache, dict) and img_filepath in self.cache:
            img = self.cache[img_filepath]
            mask = self.cache[mask_filepath]
        else:
            img = cv2.imread(img_filepath)
            # b,g,r need r channel
            mask = cv2.imread(mask_filepath)[...,-1]

        x, y = info_dict['x'], info_dict['y']
        img_width, img_height = info_dict['img_width'], info_dict['img_height']

        block_length_per_image = self.block_size
        new_img = np.full((block_length_per_image, block_length_per_image, 3), fill_value=117, dtype=img.dtype)
        new_mask = np.full((block_length_per_image, block_length_per_image), fill_value=0, dtype=mask.dtype)
        if self.test_mode:
            new_center_x = x
            new_center_y = y
        else:
            new_center_x = x + random.randint(- self.random_rect_value // 2, self.random_rect_value // 2) 
            new_center_y = y + random.randint(- self.random_rect_value // 2, self.random_rect_value // 2) 

        ori_img_left = new_center_x - block_length_per_image // 2
        ori_img_right = new_center_x + block_length_per_image // 2
        ori_img_top = new_center_y - block_length_per_image // 2
        ori_img_bottom = new_center_y + block_length_per_image // 2
        
        ori_img_left = max(0, ori_img_left)
        ori_img_top = max(0, ori_img_top)
        ori_img_right = min(ori_img_right, img_width)
        ori_img_bottom = min(ori_img_bottom, img_height)

        new_height = ori_img_bottom - ori_img_top
        new_width = ori_img_right - ori_img_left

        new_img_left = (block_length_per_image - new_width) // 2
        new_img_top = (block_length_per_image - new_height) // 2

        new_img[new_img_top:new_img_top+new_height, new_img_left:new_img_left+new_width] = img[ori_img_top:ori_img_bottom, ori_img_left:ori_img_right]
        new_mask[new_img_top:new_img_top+new_height, new_img_left:new_img_left+new_width] = mask[ori_img_top:ori_img_bottom, ori_img_left:ori_img_right]

        if isinstance(self.cache, dict) and img_filepath not in self.cache:
            self.cache[img_filepath] = img
            self.cache[mask_filepath] = mask

        if self.transform is not None:
            new_img = self.transform(new_img)

        return dict(
            image = torch.as_tensor(new_img),
            label = torch.as_tensor(np.round(new_mask / 255)).long() 
        )