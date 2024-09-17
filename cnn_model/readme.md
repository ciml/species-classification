This file guides the use of the codes in the cnn_model directory.

- utils.py and model.py bring the functions and the ResNet model used for image classification.
- sissgeo.py loads the SISSGEO dataset, with its images organized for ease of use, according to class_id.
- predict.py performs the image prediction test, returning the class most likely to be present in the image.

# Creation of the training, validation and test set.
Data Augmentation - file prepro.py
python3 prepro.py --dataset-path datasets/sissgeo

# ResNet Model Training
python3 train.py -lr --num-epochs --batch-size  --save-every 5
