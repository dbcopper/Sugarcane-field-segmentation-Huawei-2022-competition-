from typing import Optional, Tuple, Any
import torch
import glob
from pytorch_lightning import Callback, LightningModule, Trainer
import torchvision


class TensorboardOnRemoteSense(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardOnSOD
        trainer = Trainer(callbacks=[TensorboardOnSOD()])
    """
    
    def __init__(
        self,
        padding: int = 2,
        pad_value: int = 127,
        postprocess_hook = None,
        transform_hook = None
    ) -> None:
        """
        Args:
            padding: Amount of padding. Default: ``2``.
            pad_value: Value for the padded pixels. Default: ``127``.
            postprocess_hook: a hook function that post process the pred result
            transform_hook: a hook function receive two arguments (input, pred and )
        """

        super().__init__()
        self.padding = padding
        self.pad_value = pad_value
        self._should_record = False
        self.postprocess_hook = postprocess_hook if postprocess_hook is not None else (lambda x : x)
        self.transform_hook = transform_hook if transform_hook is not None else (lambda x,y : x,y)

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if not self._should_record:
            return
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(pl_module.device)
        inputs = batch['image']
        labels = batch['label']
        
        if len(labels.shape) == len(inputs.shape) and labels.size(1) == 1:
            labels = labels.expand_as(inputs)
        
        old_traing_state = pl_module.training 
        with torch.no_grad():
            pl_module.eval()
            preds = self.postprocess_hook(pl_module(inputs))
            if len(preds.shape) == len(inputs.shape) and preds.size(1) == 1:
                preds = preds.expand_as(inputs)
            pl_module.train(old_traing_state)
        el1, el2 = self.transform_hook(inputs, torch.cat([preds, labels],dim=0))
        el2, el3 = torch.split(el2, int(el1.size(0)), dim=0)
        show_tensor = torch.cat([el1.unsqueeze(dim=1), el2.unsqueeze(dim=1), el3.unsqueeze(dim=1)], dim=1)
        b,_,c,h,w = show_tensor.shape
        show_tensor = show_tensor.view(b*3, c, h ,w).to(torch.uint8)
        image = torchvision.utils.make_grid(
            tensor=show_tensor,
            nrow=3,
            padding=self.padding,
            pad_value=self.pad_value,
        )
        class_name = pl_module.__class__.__name__
        str_title = f"{class_name}_train_S/P/GT"
        trainer.logger.experiment.add_image(str_title, image, global_step=trainer.global_step, dataformats="CHW")
        self._should_record = False

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._should_record = True
        


        