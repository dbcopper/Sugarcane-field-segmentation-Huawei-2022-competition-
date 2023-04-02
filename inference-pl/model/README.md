## 在线推理

模型训练完成且编写好模型推理代码后，您可以将生成的模型导入至AI应用。

ModelArts模型包规范和配置文件说明参考
https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0055.html

本demo给出一个基本实现，选手可以根据自己需求自行更改。

模型包要求格式如下


```
OBS桶/目录名
|── inference名字
|   ├── model  必选： 固定子目录名称，用于放置模型相关文件
|   │  ├── <<自定义Python包>>  可选：用户自有的Python包，在模型推理代码中可以直接引用
|   │  ├── resnet50.pth 必选，pytorch模型保存文件，保存为“state_dict”，存有权重变量等信息。
|   │  ├── config.json 必选：模型配置文件，文件名称固定为config.json, 只允许放置一个
|   │  ├── customize_service.py  必选：模型推理代码，文件名称固定为customize_service.py, 只允许放置一个，customize_service.py依赖的文件可以直接放model目录下
```


## 使用说明

1. 将`log`中的用到的pth文件复制到`model`目录下
2. 上传到obs然后接入AI应用


## 其他

选手可以自行更改`customize_service.py`中的推理方式和数据加载方式实现更好效果，只要保证`_postprocess`函数返回的格式保持不变即可。