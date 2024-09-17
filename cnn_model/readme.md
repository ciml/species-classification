This file guides the use of the codes in the cnn_model directory.

# Creation of the training, validation and test set.
Data Augmentation - file prepro.py
python3 prepro.py --dataset-path datasets/sissgeo

# ResNet Model Training
python3 train.py -lr --num-epochs --batch-size  --save-every 5
