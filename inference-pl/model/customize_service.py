from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
import logging
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn.functional as F
from io import BytesIO
import torch.nn as nn
import torch
import json

import numpy as np

logger = logging.getLogger(__name__)
from unet import UNet
from torchvision import transforms
import os
from typing import List

IMAGES_KEY = 'input_img'
MODEL_INPUT_KEY = 'input_img'


dataset_mean = [0.19081172, 0.19135737, 0.2075335]
dataset_std = [0.07755211, 0.06542103, 0.07392063]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = dataset_mean, std = dataset_std)
])


def decode_image(file_content):

    """
    Decode bytes to a single image
    :param file_content: bytes
    :return: ndarray with rank=3
    """
    # image = base64.b64decode(base64_str)
    # image = BytesIO(file_content)
    image = Image.open(file_content)
    image = image.convert('RGB')

    return np.asarray(image)

def load_model_state_dict(pth_filepath):
    with open(pth_filepath, 'rb') as model_weight_fileobj:
        state_dict = torch.load(model_weight_fileobj, map_location='cpu')['state_dict']
        new_state_dict = type(state_dict)()
        prefix_len = len("unet.")
        for key in state_dict:
            # remove prefix "unet."
            new_state_dict[key[prefix_len:]] = state_dict[key]
        return new_state_dict


def split_image(input_image: np.ndarray, block_size:int, padding_value = 117):
    """
    input_image: numpy.ndarray with H,W,C  shape
    """
    h, w, c = input_image.shape[:3]

    new_h = (h + block_size - 1) // block_size * block_size
    new_w = (w + block_size - 1) // block_size * block_size

    padded_img  = np.full((new_h, new_w, c), fill_value = padding_value, dtype=input_image.dtype)
    padded_img[:h, :w] = input_image
    
    num_h = new_h // block_size
    num_w = new_w // block_size
    split_h_imgs = np.split(padded_img, num_h, axis = 0)

    split_imgs = list()
    for split_h_img in split_h_imgs:
        split_hw_imgs = np.split(split_h_img, num_w, axis = 1)
        split_imgs.extend(split_hw_imgs)
    
    return split_imgs, num_h, num_w


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return ",".join(map(str, rle.flatten()))


def rle_decode(rle, shape=(101,101)):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    try:
        rle = list(map(int, rle.split(",")))
        rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
        rle[:, 1] += rle[:, 0]
        rle -= 1
        mask = np.zeros([shape[0] * shape[1]], np.uint8)
        for s, e in rle:
            assert 0 <= s < mask.shape[0]
            assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
            mask[s:e] = 1
        # Reshape and transpose
        mask = mask.reshape([shape[1], shape[0]]).T
        return mask
    except:
        return np.zeros(shape, dtype=np.uint8)

  
def transform_imgs(imgs: List[np.ndarray]):
    
    img_tensors = list()
    for img in imgs:
        t = transform(img)
        # np.transpose(img, [2,0,1])
        # t = torch.as_tensor(img_tensor)
        img_tensors.append(t)
    return torch.stack(img_tensors, dim=0)

@torch.no_grad()
def split_forward(model, input_images: List[np.ndarray], block_size:int = 512):
    # 切割图像大小为 block_size
    results = list()
    device = next(model.parameters()).device
    for input_image in input_images:
        h, w = input_image.shape[:2]
        split_imgs, num_h, num_w = split_image(input_image, block_size, padding_value=117)
        input_tensors = transform_imgs(split_imgs)
        # b, 2, block_size, block_size
        infer_results = list()
        # 防止显存占用过大
        for input_tensor in input_tensors:
            infer_results.append(model(input_tensor.unsqueeze(dim=0).to(device)))
        result = torch.stack(infer_results, dim = 0)
        result = result.view(num_h, num_w, 2, block_size, block_size)
        result = torch.permute(result, dims = [2, 0, 3, 1, 4]).reshape(2, num_h * block_size, num_w * block_size)
        result = result[:, :h, :w]
        results.append(result)
    return results

#class PTServingBaseService(object):

#    def __init__(self, model_name, model_path):
#        pass

class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.model = UNet(3, 2, bilinear=True)
        state_dict = load_model_state_dict(model_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def _preprocess(self, data):

        """
        `data` is provided by Upredict service according to the input data. Which is like:
          {
              'IMAGES_KEY': {
                'filepath': b'xxx'
              }
          }
        For now, predict a single image at a time.
        """

        preprocessed_data = {}
        input_imagelist = list()
        for file_name, file_content in data[IMAGES_KEY].items():
            # print('Appending image: %s' % file_name)
            image = decode_image(file_content)
            input_imagelist.append(image)
        preprocessed_data[MODEL_INPUT_KEY] = input_imagelist
        return preprocessed_data

    def _postprocess(self, data):
        ret_dict = {
            'seg_results' : list()
        }
        results = data[MODEL_INPUT_KEY]
        # print(type(results), len(results))
        for result in results:
            result = torch.argmax(result, dim=0, keepdim=False)
            result_numpy = result.detach().cpu().numpy()
            # print('result numpy', result_numpy.dtype, result_numpy.shape, result_numpy.min(), result_numpy.max())
            result_rle = rle_encode(result.detach().cpu().numpy())
            ret_dict['seg_results'].append({
              'shape':tuple(result.shape),
            	'rle_code':result_rle
            })
        return ret_dict

    def _inference(self, data):
        result = {}
        for k, v in data.items():
            result[k] = split_forward(self.model, v)
        return result




if __name__ == "__main__":
    import base64

    img_path = r'./test.png'
    # img_path = r'D:\广西数据\baseline-code\1.jpg'
    with open(img_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)
        # base64_data = base64_data.decode()

    service = PTVisionService('model_name', './unet-epoch=00-val_miou_epoch=0.505.pth')
    preprocessed_data = service._preprocess({
        IMAGES_KEY: {
            img_path : BytesIO(base64.b64decode(base64_data))
        }
        
    })
    inferenced_data = service._inference(preprocessed_data)
    postprocessed_data = service._postprocess(inferenced_data)

    print(postprocessed_data)


