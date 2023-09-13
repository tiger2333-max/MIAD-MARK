#Dataset Availability

Three representatively datasets are selected in our paper: 'Messidor'(Fundus), 'ISIC2019'(Skin Lesion), and 'BTMRI'(Brain Tumor MRI).

These open access datasets can be downloaded from:

    'Messidor': https://www.adcis.net/en/third-party/messidor
    
    'ISIC2019': https://challenge.isic-archive.com/data/#2019
    
    'BTMRI': https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679

The download path is set to './Model Training/data'.

Uncompress dataset to './Model Training/data/[Your Dataset Name]'.

#Dataset Configuration
You can edit the class name in "./Model Training/dataset.py", and calculate the mean value and the standard deviation of the training set.

#Dataset Split
The training set and validation set can be splited through './Model Training/data split.py'.
Here we supply an example of "messidor" dataset. The training set sampled as category distribution without replacement.
Please note the dataset with unbalanced category distributions, eg. 'ISIC2019'. In that situation, the minority category needs to be oversampled while the majority category needs undersampling.
Besides, we crop the black boder of images in order to computational efficiency.

Notion: each dataset annotations and image format might be different. If you want to use our training process, please split dataset and write the image path and label into the '.txt' file as the './Model Training/data split.py'.

#Model Training
Running settings can be modified in './Model Training/argments.py'. Standard DNN models can be trained by simply run this line:

    cd './Model Training'
    
    python train.py --root_path --model_name --dataset --cla_num --max_epoch --bs --lr_init --optimizer --loss --lr_step --lr_decay --seed

The loss curves (Tensor board), model weights and confusion matrixes will be recorded and saved in './Model Training/data/[Your Dataset Name]/result'. 

Our pretrained model weights can be accessed from Google Drive:

    https://drive.google.com/drive/folders/1z-Fj86R-lqAjvyXm3m51P-eimETqk92T
    
    

#Utilization
Other utilizations can be found in './Model Training/utils.py'. There you can add new models to the "get_model" function.
