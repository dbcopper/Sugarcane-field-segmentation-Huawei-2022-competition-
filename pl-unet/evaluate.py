import os
import glob
import sys
import pandas as pd
import numpy as np
import time
import moxing as mox
import json

os.system("pip install torch")
os.system("pip install opencv-python-headless")

parser = argparse.ArgumentParser()
# 输入批处理的结果目录(模型预测结果目录)
parser.add_argument('--data_url', default='s3://your_obs_url', help='dir of dataset')  
# 输出判分结果的目录
parser.add_argument('--train_url', default='s3://your_obs_url', help='dir of output')  
# 容器本地的临时目录
parser.add_argument('--local_url', default='/cache', help='dir of output')  
# 测试数据标签文件的目录
parser.add_argument('--ground_truth_url', default='s3://your_obs_url',help='dir of ground_truth')  
args, _ = parser.parse_known_args()

def eval(gt_dir, pred_dir):
    from metrics.evaluation.metrics import mean_iou
    import cv2
    import torch

    gt_suffix = '_mask.png'
    pred_suffix = '_img.png_result.txt'
    gt_filepaths = glob.glob(os.path.join(gt_dir, "*" + gt_suffix))

    def gen_pred_path(gt_path):
        return os.path.join(pred_dir, os.path.basename(gt_path[:-len(gt_suffix)] + pred_suffix))

    count = len(gt_filepaths)
    iou = torch.zeros((2, ))
    for gt_filepath in gt_filepaths:
        pred_filepath = gen_pred_path(gt_file)
        if not os.path.isfile(pred_file):
            continue
        gt_img_obj = cv2.imread(gt_filepath)[...,-1]
        if gt_img_obj is None:
            count -= 1
            print('GT file %s is bad' % gt_filepath, file=sys.stderr)
            continue
        
        with open(pred_filepath) as f:
            pred_json = json.load(f)
        # N,2 (x,y)
        pred_axis = pred_json['seg_results'][0]
        xs, ys = np.split(np.array(pred_axis), 2, axis=1)
        xs = xs[:, 0]
        ys = ys[:, 0]

        pred_img_obj = np.zeros_like(gt_img_obj)
        pred_img_obj[ys, xs] = 255

        pred_img_obj /= 255
        gt_img_obj /= 255
        
        results = mean_iou(pred_img_obj, gt_img_obj, 2, 255, nan_to_num=0.0)
        iou += results['IoU']
    
    return float(iou.mean() / count)

# 定义本地的路径
local_data_path = os.path.join(args.local_url, 'data')  # local path for data files
local_label_path = os.path.join(args.local_url, 'label')  # local path for label files
local_save_path = os.path.join(args.local_url, 'save')  # local path for saved score

time_start = time.time()
# 把OBS的数据拷贝到容器本地
mox.file.copy_parallel(args.data_url, local_data_path)
mox.file.copy_parallel(args.ground_truth_url, local_label_path)

score = statistic_precision(local_label_path, local_data_path) # gt and predict
results = pd.DataFrame({'mIoU': [round(score,4)]})

# 将统计结果写到本地文件
local_output_file = os.path.join(local_save_path, 'results.csv')  # 所有大赛的评分结果都默认保存为results.csv
if not os.path.exists(local_save_path):
    os.mkdir(local_save_path)
results.to_csv(local_output_file, header=None, index=False)
mox.file.copy_parallel(local_save_path, args.train_url)
time_end = time.time()
print('time cost:', time_end - time_start, 's')

