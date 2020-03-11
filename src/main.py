import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['IMG_HEIGHT']='137'
os.environ['IMG_WIDTH']='236'
os.environ['EPOCHS']='50'
os.environ['TRAIN_BATCH_SIZE']='64'
os.environ['TEST_BATCH_SIZE']='8'
os.environ['MODEL_MEAN']='(0.485, 0.456, 0.406)'
os.environ['MODEL_STD']='(0.229, 0.224, 0.225)'
os.environ['BASE_MODEL']='resnet34'
os.environ['TRAINING_FOLDS_CSV']='../input/train_folds.csv'
os.environ['IMAGE_PKL_PATH']='../input/image_pickles'

if __name__ == '__main__':
    from train import main
    train_folds=(0, 1, 2, 3)
    valid_folds=(4,)
    main(train_folds, valid_folds)

# export TRAINING_FOLDS="(0, 1, 2, 3)"
# export VALIDATION_FOLDS="(4,)"
train_folds=(0, 1, 2, 3)
valid_folds=(4,)
main(train_folds, valid_folds)
# /opt/anaconda3/envs/torch/bin/python3 train.py

# export TRAINING_FOLDS="(0, 1, 2, 4)"
# export VALIDATION_FOLDS="(3,)"
# /opt/anaconda3/envs/torch/bin/python3 train.py

# export TRAINING_FOLDS="(0, 1, 4, 3)"
# export VALIDATION_FOLDS="(2,)"
# /opt/anaconda3/envs/torch/bin/python3 train.py

# export TRAINING_FOLDS="(0, 4, 2, 3)"
# export VALIDATION_FOLDS="(1,)"
# /opt/anaconda3/envs/torch/bin/python3 train.py

# export TRAINING_FOLDS="(4, 1, 2, 3)"
# export VALIDATION_FOLDS="(0,)"
# /opt/anaconda3/envs/torch/bin/python3 train.py

