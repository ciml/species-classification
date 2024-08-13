import numpy as np
from PIL import Image
import os
import imageio

"""
Loader of SISS-Geo images

   The dataset of SISS-Geo images is organized as follows:

       sissgeo
       ├── test
       │   └── 1276-2-1.jpg
       └── train
           ├── 1234-7-1.jpg
           ├── 1234-7-2.jpg
           └── 1235-3-1.jpg

   Each SISS-Geo image obeys the pattern <id>-<class>-<n>.jpg, where 'id' is the
   register identification number, 'class' is the class index (animal type) and
   'n' the n-th image of the register.
"""

def load_train_test( dataset_path, type ):
   # Lists all image names in dataset_path/type, which type being either `train` or `test`
   list_of_images = os.listdir( "%s/%s" % ( dataset_path, type ) )
   n = len( list_of_images )

   x = np.ndarray(shape = (n,32,32,3), dtype = np.uint8)  # n -> actual number of images in the directory
   y = np.ndarray(shape = (n), dtype = np.uint8)  # n -> actual number of images in the directory
   for each image_name in list_of_images:
      # Load each image (from image_name) in the dir as a 3D ndarray
      x[i] = imageio.imread(image_name)
      # Extract the class index from image_name
      y[i] = int( image_name.split('-')[1] )

   return x, y


def load_sissgeo(dataset_path, shape):
   """
       Load the SISS-Geo image dataset

       Parameters
       ----------
       dataset_path : str
       path to the dataset files

       shape : tuple
       target shape of the loaded instances

       Returns
       -------
       x_train : np.array
           training instances
       y_train : np.array
           training labels
       x_test : np.array
           testing instances
       x_test : np.array
           testing labels
   """

   x_train, y_train = load_train_test( dataset_path, "train" )
   x_test, y_test = load_train_test( dataset_path, "test" )

   return x_train, y_train, x_test, y_test
