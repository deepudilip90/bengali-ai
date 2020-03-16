import pandas as pd
import os
from tqdm import tqdm
import h5py
import numpy as np
from glob import glob



def create_image_hd5(input_path, output_path):

    parquet_files = glob(input_path + 'train_*.parquet')
    parquet_files.sort()

    hd5file = h5py.File(output_path + 'image_dataset.h5", "w")

    for file in parquet_files:
        print('Now Reading from file: ', file)
        df_parquet = pd.read_parquet(file)
        image_ids = df_parquet.image_id
        image_arrays = df_parquet.drop('image_id', axis=1).values
        for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            image = image_arrays[idx,:].reshape(236, 137)
            hd5file.create_dataset(image_id, np.shape(image), h5py.h5t.STD_U8BE, data=image)

    hd5file.close()

if __name__ == '__main__':
    ## Read parquet images and store to hdf5
    parquet_dir = '/Users/deepudilip/ML/Kaggle /benagli-ai/input/'
    hd5_file_path = '/Users/deepudilip/ML/Kaggle /benagli-ai/input/'
    create_image_hd5(input_path, output_path)
        
# labels_df = pd.read_csv('/Users/deepudilip/ML/Kaggle /benagli-ai/input/train.csv')
# label_arrays = labels_df.drop(columns=['image_id', 'grapheme']).values
# images = np.asarray(images)

        
#         labels[image_id] = label_arrays