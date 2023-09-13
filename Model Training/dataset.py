import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argments import args

def get_mean_std(args):
    dataset = args.dataset

    # Define a dictionary to store mean and std values for different datasets.
    # If you use your own dataset, please calculate the relative mean and std.

    dataset_stats = {

        #Fundus
        'messidor': {
            'mean': [0.5234, 0.2478, 0.0824],
            'std': [0.3086, 0.1577, 0.0699]
        },
        'IDRiD': {
            'mean': [0.4566, 0.3154, 0.2196],
            'std': [0.2554, 0.1859, 0.1547]
        },
        'kaggle2015': {
            'mean': [0.4571, 0.3142, 0.2173],
            'std': [0.2518, 0.1811, 0.1492]
        },

        #Skin Lesions
        'isic2019': {
            'mean':[0.6565, 0.5199, 0.5147],
            'std':[0.2380, 0.2128, 0.2234]
        },

        #Brain Tumor MRI
        'BTMRI': {
            'mean':0.2236,
            'std':0.1608
        }
    }

    # Check if the dataset is recognized
    if dataset not in dataset_stats:
        raise ValueError(f"Dataset '{dataset}' not recognized.")

    mean = dataset_stats[dataset]['mean']
    std = dataset_stats[dataset]['std']

    return mean, std



def cal_mean_std(loader):

    '''
    Calculate mean and std of dataset
    input: Dataloader()
    output: Mean: tensor([, ,])
            Std: tensor([, , ])
    '''

    data_sum = 0
    data_squared_sum = 0
    num_batches = len(loader)

    for data, _ in loader:
        # data: [batch_size, channels, height, width]

        data_sum += torch.mean(data, dim=(0, 2, 3)) # [channels]
        data_squared_sum += torch.mean(data ** 2, dim=(0, 2, 3))  # [channels]

    mean = data_sum / torch.tensor(num_batches)

    std = torch.sqrt(data_squared_sum / torch.tensor(num_batches) - mean ** 2)

    return mean, std

classname_dict = {
    # Fundus
    'messidor': ('No DR', 'Mild', 'Moderate', 'Severe'),
    'IDRiD': ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'),

    # Skin Lesions
    'isic2019': ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'),

    # Brain Tumor MRI
    'BTMRI': ('meningioma', 'glioma', 'pituitary tumor')
}





if __name__=='__main__':

    print('Here is a test code of dataset.py')

    ##Here is an example of calculate mean and std of Cifar-10 dataset

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    #
    # #train set
    # dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    #
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
    #
    # mean, std = cal_mean_std(loader)
    #
    # print("Mean:", mean)
    # print("Std:", std)




    # print(args.dataset,get_mean_std(args))