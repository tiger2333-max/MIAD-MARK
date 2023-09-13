import os
from argments import args
import pandas as pd
from utils import crop_black_border
import cv2
from sklearn.model_selection import train_test_split

def data_split(df,target,seed,sample_frac):
    grouped = df.groupby(target)
    data_by_category = {category: group for category, group in grouped}
    print(data_by_category)

    train_data = []
    test_data = []

    for category, data in data_by_category.items():

        # sampled in category
        sampled_data = data.sample(frac=sample_frac, random_state=seed)
        test_data.append(data.drop(sampled_data.index))
        train_data.append(sampled_data)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    return train_df,test_df


if __name__=='__main__':

    #dataset=args.dataset
    # data_dir = './Model Training/data' #put dataset into 'data_dir/dataset/image'
    dataset = 'messidor'
    data_dir = 'E:/train_DR/data'

    if dataset == 'messidor':
        annotation_file_path = os.path.join(data_dir,dataset,'annotations')
        annotation_file_list = os.listdir(annotation_file_path)
        print(annotation_file_list)

        total_df = pd.DataFrame()

        #create dataframe for all .xls file
        for xls_name in annotation_file_list:
            xls_path = os.path.join(annotation_file_path, xls_name)
            anno_df = pd.read_excel(xls_path)
            total_df = pd.concat([total_df, anno_df],ignore_index=True)

        total_df = pd.concat([total_df['Image name'], total_df['Retinopathy grade']], axis=1)
        value_counts = total_df ['Retinopathy grade'].value_counts()

        #crop image black border

        crop_df = pd.DataFrame(columns=['cropped image path', 'label'])
        cropped_image_path = []

        for i in range(0,20):
        #for i in range(len(total_df['Image name'])):
            img_path = os.path.join(data_dir,dataset,'image',total_df['Image name'][i])

            output_dir = os.path.join(data_dir, dataset, 'cropped_image')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
            cropped_image_path.append(output_path)

            if os.path.exists(output_path):
                print(f"File already exists at {output_path}. Skipping saving.")

            else:
                crop_img = crop_black_border(img_path)
                cv2.imwrite(output_path, crop_img)

        crop_df['cropped image path'] =  cropped_image_path
        crop_df['label'] = total_df['Retinopathy grade']
        # print(crop_df)

        # train-test split
        train_df, test_df = data_split(crop_df,'label',seed=args.seed,sample_frac=0.75)
        train_df.to_csv(os.path.join(data_dir,dataset,'train.txt'), sep='\t', index=False, header=False)
        test_df.to_csv(os.path.join(data_dir,dataset,'test.txt'), sep='\t', index=False, header=False)






