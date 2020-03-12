import pandas as pd
import joblib
import glob 
from tqdm import tqdm
import os

def create_image_pickles(parquet_path, output_pickle_path):
    files = glob.glob(parquet_path)
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop('image_id', axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            output_file_name = output_pickle_path + os.sep+ f"{img_id}.pkl"
            # joblib.dump(image_array[j, :], output_file_name)


if __name__ == '__main__':
    
    create_image_pickles('../input/train_*.parquet', '../input/image_pickles')
