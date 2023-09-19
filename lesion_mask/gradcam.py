import torch
import numpy as np
import pandas as pd
from PIL import Image
import os.path
from Model_Training.utils import get_model
from lesion_mask.utils import get_target_layer,mkdir,GradCAM,GuidedBackPropagation
import cv2
from skimage import io
from configs import args
from Model_Training.dataset import get_mean_std

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    CAM
    :param image: [H,W,C]
    :param mask: [H,W], (0,1)
    :return: tuple(cam,heatmap)
    """
    # turn mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # add heatmap on image
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def gen_gb(grad):
    """
    guided-back propagation
    :param grad: tensor,[3,H,W]
    :return:
    """
    #
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.png'.format(prefix, network, key)), image)

def prepare_input(image,mean,std):
    image = image.copy()
    means = np.array(mean)
    stds = np.array(std)
    image = (image - means) / stds
    image = np.float32(image)
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]
    return torch.tensor(image, requires_grad=True)

if __name__ == '__main__':

    dataset = args.dataset
    root_path = args.root_path
    model_name = args.model_name
    cla_num = args.cla_num
    seed = args.seed

    mean, std = get_mean_std(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(root_path, 'models', args.dataset, model_name + '.pkl')
    print(model_path)

    model = get_model(model_name, cla_num, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    #get the targeted layer for grad-cam
    conv_list = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_list.append(name)
    print(conv_list)

    target_layers = get_target_layer(model, model_name)
    print(target_layers)


    # set proper numbers of images, the following process may cause out of the GPU memory.
    df = pd.read_csv(os.path.join(root_path, 'data', dataset, 'train.txt'),
                     sep=' ', names=['img', 'label'])

    sample_index = list(df.index)

    if args.sample_num is not False:

        sample_df = df.groupby('label').apply(lambda x: x.sample(args.sample_num, random_state=seed))
        sample_index = list(sample_df.index)

    for i in sample_index:
        img_path = df['img'][i]
        img = Image.open(img_path).convert('RGB')

        img_np = np.array(img)
        size = 224
        if args.model == 'inceptionv3':
            size = 299
        img_resize = np.float32(cv2.resize(img_np, (size, size))) / 255
        print(np.max(img_resize))

        inputs = prepare_input(img_resize, mean, std)

        with torch.no_grad():
            pred = model(inputs.to(device))

        print(pred)
        pred_cla = torch.argmax(pred)
        print('pred_cla:', pred_cla)
        label = int(df['label'][i])
        print('label', label)

        if pred_cla == label:
            layer_name = target_layers
            image_dict = {}
            print('layer name', layer_name)

            # grad_cam
            grad_cam = GradCAM(model, layer_name)
            mask = grad_cam(inputs, label)
            image_dict['cam'], image_dict['heatmap'] = gen_cam(img_resize, mask)
            grad_cam.remove_handlers()
            mask_gray = (mask * 255).astype(np.uint8)

            # GuidedBackPropagation
            gbp = GuidedBackPropagation(model)
            inputs.grad.detach_()
            grad = gbp(inputs)
            gb = gen_gb(grad)
            image_dict['gb'] = norm_image(gb)

            # Guided Grad-CAM
            cam_gb = gb * mask[..., np.newaxis]
            image_dict['cam_gb'] = norm_image(cam_gb)

            save_dir = os.path.join(root_path, 'lesion_mask', 'results', dataset, model_name, layer_name)
            mkdir(save_dir)
            io.imsave(
                os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + '_' + str(label) + '.png'),
                (img_resize * 255).astype(np.uint8))  # save the original image after resize

            save_image(image_dict, os.path.basename(img_path), model_name, save_dir) #save cam,gb and cam_gb
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[
                0] + '-' + model_name + '-mask' + '.png'), mask_gray) #save gray-scale mask


