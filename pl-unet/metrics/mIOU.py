import sys
import os
import torch
from torchmetrics import Metric
from typing import Any, Optional, Tuple, List, Union
from .evaluation import mean_iou

class mIOU(Metric):

    full_state_update  = False
    
    def __init__(
        self,
        num_classes,
        ignore_index,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,        
        nan_to_num=None,
        label_map=dict(),
        reduce_zero_label=False
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.reduce_zero_label = reduce_zero_label
        self.label_map = label_map
        self.add_state("iou", default=torch.tensor([0.0, ] * num_classes), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model, shape: B,C,H,W, C==num_class
            target: Ground truth values, shape: B,H,W
        """
        preds = torch.argmax(preds, dim=1)
        for pred, gt in zip(preds, target):
            ret_metrics = mean_iou(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), self.num_classes, self.ignore_index, nan_to_num=self.nan_to_num, label_map=self.label_map, reduce_zero_label=self.reduce_zero_label)
            iou = ret_metrics['IoU']
            self.iou += torch.as_tensor(iou).to(self.iou.device)
        self.count += preds.size()[0]

    def compute(self) -> torch.Tensor:
        """
        Computes mae over state.
        """
        if self.count == 0:
            return torch.zeros_like(self.count)
        return self.iou.mean() / self.count


if __name__ == '__main__':

    import numpy as np
    pred_size_results = (10, 10, 30, 30)
    pred_size_label = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.uniform(0, num_classes, size=pred_size_results)
    label = np.random.randint(0, num_classes, size=pred_size_label)

    results_torch = torch.as_tensor(results).float()
    label_torch = torch.as_tensor(label).long()
    mean_iou_metric = mIOU(num_classes=num_classes, ignore_index=ignore_index)
    c = mean_iou_metric(results_torch, label_torch)
    print(c)

    d = mean_iou(np.argmax(results, axis=1), label, num_classes, ignore_index)
    d_iou = d['IoU'].mean()
    print(d_iou)

    mean_iou_metric.reset()

    mean_iou_metric(results_torch, label_torch)
    mean_iou_metric(results_torch, label_torch)
    e = mean_iou_metric.compute()
    print(e)

