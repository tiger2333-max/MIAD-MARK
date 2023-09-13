import torch.nn as nn
import torchvision.models as models
from transformers import ViTForImageClassification
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

def get_model(model_name, cla_num, device, pretrained=True):

    """
    Create and configure a neural network model based on the specified model name.

    Args:
        model_name (str): Name of the model to use ('resnet18', 'resnet50', 'vgg16', etc.).
        cla_num (int): Number of classes for the final classification layer.
        pretrained (bool): Whether to use pre-trained weights (if available).

    Returns:
        nn.Module: The configured neural network model.

    Example usage:
        model = get_model('resnet18', 10, 'cuda')

    """

    model_dict = {
        'resnet18': (models.resnet18, 512),
        'resnet50': (models.resnet50, 2048),
        'vgg16': (models.vgg16, 4096),
        'inceptionv3': (torch.hub.load, ('pytorch/vision:v0.10.0', 'inception_v3', True)),
        'mobilenetv3': (models.mobilenet_v3_large, 1280),
        'densenet121': (models.densenet121, 1024),
        'vit': (ViTForImageClassification.from_pretrained, ('google/vit-base-patch16-224',)),
    }

    create_model, fc_in_features = model_dict.get(model_name, (None, None))

    if create_model is None:
        raise ValueError(f"Model '{model_name}' not recognized.")

    net = create_model(pretrained=pretrained)
    if model_name != 'vit':
        net.fc = nn.Linear(fc_in_features, cla_num)
    else:
        net.classifier = nn.Linear(768, cla_num)

    return net.to(device)


def crop_black_border(image):

    """
    Crop the black border of the image.
    Args:
        image: The input image (OpenCV image object).

    Returns:
        The cropped image (OpenCV image object).

    Note:
        Few images' black border may still exist(completely or partially) after this process.
        We suggest you properly adjust parameters of Median filter or binary threshold.

    """

    if not os.path.isfile(image):
        raise FileNotFoundError(f"Image file not found: {image}")

    img = cv2.imread(image)
    if img is None:
        raise ValueError(f"Failed to read image: {image}")

    img = cv2.medianBlur(img, 5)  # Median filter for denoising
    binary_image = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    x, y = binary_image.shape

    print(x,y)
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x or not edges_y:
        raise ValueError("Failed to find edges in the image")

    bottom = min(edges_y)
    top = max(edges_y)
    left = min(edges_x)
    right = max(edges_x)

    w = right - left
    h = top - bottom

    crop_image = img[left:left + w, bottom:bottom + h]

    return crop_image




def validate(net, data_loader, set_name, classes_name,model_name, device):
    """
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'test
    :param classes_name:
    :return: Confusion Matrix and Accuracy
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        images, labels = data[0].to(device),data[1].to(device)
        images = Variable(images)
        labels = Variable(labels)

        outputs = net(images)
        print(outputs)
        if model_name == 'vit':
            outputs = net(images).logits
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)


        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                                     conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_confMat(confusion_mat, classes, set_name, out_dir):

    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()


    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

 
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

 
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
  
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.close()


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor



class MyDataset(Dataset):

    def __init__(self, txt_path, transform=None, target_transform=None):

        fh = open(txt_path, 'r')
        imgs = []

        for line in fh:
            line = line.rstrip()
            words = line.split()
            print(line,words)
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
