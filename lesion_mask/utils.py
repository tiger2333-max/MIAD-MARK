import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import cv2
from configs import args

def sampling_for_attack(args):
    root_path = args.root_path
    dataset = args.dataset
    sample_num = args.sample_num
    cla_num = args.cla_num
    seed = args.seed

    txt_path = os.path.join(root_path, 'datasets', dataset, dataset + '.txt')
    df = pd.read_csv(txt_path, sep=' ', names=['img', 'label'])
    sample_df = df.groupby('label').apply(lambda x: x.sample(sample_num, random_state=seed))
    sample_index = list(sample_df.index)
    sample_index_list=[]
    for i in range(0,len(sample_index)):
        sample_index_list.append(sample_index[i][1])
    print('sample_index', sample_index_list, '\n', 'sample num', sample_num * cla_num)
    return sample_index_list

def get_target_layer(model,model_name):
    conv_list = []
    target_layers=[]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_list.append(name)

    if model_name=='resnet50':
        layer_index = [i for i in range(len(conv_list)) if "conv3" in conv_list[i]]
        target_layers = [conv_list[i] for i in layer_index][-1]

    elif model_name=='vgg16':
        target_layers=conv_list[-1]

    elif model_name=='densenet121':
        layer_name = 'transition'
        layer_index = [i for i in range(len(conv_list)) if str(layer_name) in conv_list[i]]
        layer_index = [x - 1 for x in layer_index] + [-1]
        target_layers = [conv_list[i] for i in layer_index][-1]

    elif model_name=='inceptionv3':
        target_layers=['Mixed_5d.branch_pool.conv','Mixed_6e.branch_pool.conv','Mixed_7c.branch_pool.conv'][-1]

    elif model_name=='mobilenetv3':
        target_layers=['features.15.block.0.0', 'features.15.block.1.0', 'features.15.block.3.0', 'features.16.0'][-1]

    return target_layers

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAM:
    def __init__(self, net, layer_name):
        """
        Initialize GradCAM object.

        Args:
            net (nn.Module): The neural network.
            layer_name (str): The name of the target layer for GradCAM.
        """
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):

        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index,sampling_rate=1.0):

        """
         Generate GradCAM heatmap for the given input image.

         Args:
             inputs (torch.Tensor): Input image tensor.
             index (int): Class index for which to generate the heatmap.
             sampling_rate (float): Sampling rate for selecting channels.

         Returns:
             np.ndarray: GradCAM heatmap.
         """

        self.net.zero_grad()
        output = self.net(inputs.to(device))  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        n_channels = feature.shape[0]  # get features of the channels
        sampled_n_channels = int(n_channels * sampling_rate)  # sampled channels
        sampled_indices = np.argsort(weight)[-sampled_n_channels:]  # choose the top k channels
        cam = feature[sampled_indices, :, :] * weight[sampled_indices, np.newaxis, np.newaxis]  # [k,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # normalize
        cam -= np.min(cam)
        cam /= np.max(cam)

        # resize to 224*224
        size=224
        if args.model=='inceptionv3':
            size=299
        cam = cv2.resize(cam, (size,size))
        return cam


class GuidedBackPropagation:
    def __init__(self, net):
        """
        Initialize GuidedBackPropagation object.

        Args:
            net (nn.Module): The neural network.
        """
        self.net = net
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self._backward_hook)

    @staticmethod
    def _backward_hook(module, grad_in, grad_out):
        """
        Modify gradients during backward pass.

        Args:
            module: ReLU module.
            grad_in: Tuple of input gradients.
            grad_out: Tuple of output gradients.

        Returns:
            Tuple of modified input gradients.
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=None):
        """
        Generate guided backpropagation gradient for the given input image.

        Args:
            inputs (torch.Tensor): Input image tensor.
            index (int): Class index for which to generate the gradient.

        Returns:
            torch.Tensor: Guided backpropagation gradient.
        """
        torch.cuda.empty_cache()
        self.net.zero_grad()
        output = self.net(inputs.to(device))  # [1, num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()
        return inputs.grad[0]  # [3, H, W]
