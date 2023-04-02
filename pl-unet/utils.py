import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms, p = 1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, label):
        if np.random.rand() < self.p: 
            for t in self.transforms:
                img, label = t(img, label)
            return img, label
        else:
            return img, label

class BGR2RGB:
    def __init__(self, dim = 1):
        self.dim = dim
    
    def __call__(self, img, label):
        return torch.flip(img, dims=[self.dim]), label

class DeNormalizeAndShow:
    def __init__(self, mean, std, num_classes):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.num_classes = num_classes
        # 填充color_map
        copper = cm.get_cmap('copper', num_classes)
        self.color_map = np.array([item[:3] for item in copper(range(num_classes))])

    def __call__(self, image, mask):
        image_len = len(image.shape)

        mean_tensor_shape = [1,] * max(0, image_len - 3) + [self.mean.size,]
        mean_tensor_shape.extend([1,] * (image_len - len(mean_tensor_shape)))

        std_tensor_shape = [1,] * max(0, image_len - 3) + [self.std.size,]
        std_tensor_shape.extend([1,] * (image_len - len(std_tensor_shape)))

        image = (image * torch.from_numpy(self.std).type_as(image).to(image.device).view(std_tensor_shape))
        image += torch.from_numpy(self.mean).type_as(image).to(image.device).view(mean_tensor_shape)
        image *= 255.0
        mask = mask.long()

        color_map = torch.as_tensor(self.color_map).view(self.num_classes, 3).to(image.dtype).to(image.device)
        mask = color_map[mask]
        
        dims = list(range(len(mask.shape)))
        last_dim = dims.pop(-1)
        dims.insert(1, last_dim)
        mask = mask.permute(dims = dims)
        if len(mask.shape) >= 3:
            mask = mask.squeeze(dim=2)

        mask = mask.to(image.dtype) * 255.0
        return image, mask


if __name__ == "__main__":
    num_classes = 20
    
    denormalize = DeNormalize([0.5,0.5,0.5], [0.5,0.5,0.5], num_classes)
    image = torch.rand(1,3,224,224)
    mask = torch.randint(0, num_classes, (2,1,224,224)).long()
    
    new_image, new_mask = denormalize(image, mask)
    print(new_image.shape, new_image.dtype)
    print(new_mask.shape, new_mask.dtype)
    
    bgr2rgb = BGR2RGB(dim=-1)
    a = torch.rand(1,2,2, 3)
    ret = bgr2rgb(a, a)
    print(ret)