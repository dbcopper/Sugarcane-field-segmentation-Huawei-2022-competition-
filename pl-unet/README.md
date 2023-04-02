

## 一、安装说明

参考 https://pytorch.org/get-started/locally/ 安装最新版本的pytorch后(推荐torch1.11.0)，执行以下命令

```
pip install opencv-python
pip install matplotlib
pip install pytorch-lightning
pip install torchmetrics
```

## 二、训练


1. 更改train.py中数据集路径为正确路径

    > Note:选手需要自行划分数据集为训练集和验证集

2. 在当前目录运行，当前是设置gpu个数为2，用户可以根据自己的情况去更改
    ```
    python train.py
    ```

3. 学习率、batchsize等参数可以直接在train.py中更改，也可以按照如下命令配置yaml文件  
    3.1 yaml文件示例
    ```yaml
    lr: 0.0001
    ```
    3.2 训练命令
    ```bash
    python train.py -yaml yaml文件地址
    ```

4. 关于pytorch-lightning框架的更多用法参考  https://pytorch-lightning.readthedocs.io/


## 三、可视化
默认log文件在`log`文件夹下面，找到`tb_logs/rs_unet`然后找到对应运行版本

```bash
tensorboard --logdir 文件夹地址
```

## 四、其他

当前baseline提供了较为全面的功能支持，选手可以使用当前baseline作为一个框架去扩展思路，如果参赛选手在使用过程中有任何疑问可以在群里交流。