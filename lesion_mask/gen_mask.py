import os
import cv2
from lesion_mask.utils import mkdir
from configs import args

layer_names = {
    'messidor': {'resnet50': 'layer4.2.conv3', 'vgg16': 'features.28', 'mobilenetv3': 'features.16.0',
                 'inceptionv3': 'Mixed_7c.branch_pool.conv', 'densenet121': 'features.denseblock4.denselayer16.conv2'},
    'isic2019': {'resnet50': 'layer4.2.conv3', 'vgg16': 'features.28', 'mobilenetv3': 'features.16.0',
                 'inceptionv3': 'Mixed_7c.branch_pool.conv', 'densenet121': 'features.denseblock4.denselayer16.conv2'},
    'nih_chestxray': {'resnet50': 'layer4.2.conv3', 'densenet121': 'features.denseblock4.denselayer16.conv2'},
    'brain_tumor': {'resnet50': 'layer4.2.conv3', 'vgg16': 'features.28', 'mobilenetv3': 'features.16.0',
                    'inceptionv3': 'Mixed_7c.branch_pool.conv',
                    'densenet121': 'features.denseblock4.denselayer16.conv2'},
}


if __name__=='__main__':

      dataset = args.dataset
    root_path = args.root_path
    model_name = args.model_name
    cla_num = args.cla_num
    seed = args.seed
    down_val = args.down_val
    up_val = args.up_val
    step= args.step
    layer_name = layer_names[dataset][model_name]

    mask_dir=os.path.join(root_path, 'lesion_mask', 'results', dataset, model_name, layer_name)
    gradcam_list=os.listdir(mask_dir)
    mask_index = [i for i in range(len(gradcam_list)) if 'mask' in gradcam_list[i]]
    mask_list=[gradcam_list[i] for i in mask_index]
    print(mask_list)
    save_path = mask_dir + '_threshold_mask'
    mkdir(save_path)
    print(save_path)
    for mask in mask_list:
        mask_img=cv2.imread(os.path.join(mask_dir,mask),0)
        print('mask_img',mask)
        for threshold in range(down_val,up_val,step):
            ret, mask_new = cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(save_path,os.path.splitext(mask)[0]+'_threshold_'+str(threshold)+'.png'),mask_new)+'.png'),mask_new)
