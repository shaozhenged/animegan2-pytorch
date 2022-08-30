# 参考资料
# https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
# https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html

import torch
import torchvision
import torch.onnx

from model import Generator

model_path = './weights/face_paint_512_v2.pt'

#使用预训模型初始化模型
net = Generator()
net.load_state_dict(torch.load(model_path))
net.train(False)

x = torch.rand(1,3,1024,1024)

torch_out = torch.onnx._export(net, x, "face_paint_512_v2.onnx", export_params=True, opset_version=11)