This file details the Sissgeo dataset, as well as the data splitting used to run the classification models.

The initial dataset consists of images of 15 animal classes. The set is separated into 3 partitions:
- 70% for the Training set;
- 15% for the Validation set;
- 15% for the Test set.

# Training Set
- Unbalanced set containing images from the 15 classes under study;
- Data augmentation is performed to generate 10,000 images from each class;
- From the expanded data set, the partition is performed into other smaller sets, to analyze the functioning of ResNet in classifying the images.

# Partitions in the Training Set
- Initially, 1,000 images of each class are used for classification;
- After that, the number of images used for training is progressively increased: 2,000 images, 3,000 images, 5,000 images, 7,000 images and 10,000 images.
- The accuracy obtained in ResNet training for each number of images is analyzed.

# Validation and Test Set
- Initially, the data augmentation operations used in the model's training images are replicated;
- An initial test seeks to use the augmented images and maintain the initial proportion between the training, validation and test sets;
- The images without data augmentation are also used.

# General Observations
- The records used for training, validation and testing are previously separated, in order to use the same records for training the species distribution model.
- There are records that have more than one image. In this case, all available images are used and the classification is done for each record in two ways: the record is considered, averaging the predicted probability for all images, or each image and its predicted probability are used separately.
- In MaxEnt, the result presents an estimate of the existence of the species based on its geographic location. This normalizes the generated results for comparison purposes.
- The original images are present in all training datasets.
- Images are randomly selected from the training set partitions. However, we aim to ensure a balance between the number of enlarged images for each record.
