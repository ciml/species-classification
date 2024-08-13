#!/usr/bin/env python3

"""augmentate
"""

import os
import sys
import argparse
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

def rotate():
   print("Calling rotate method with arg {}".format(args.rotate))
   image = imageio.imread(args.image)
   rotate = iaa.Affine(rotate=(args.rotate))
   image_aug = rotate(image=image)

   outputfilename = args.image[:-4] + "-rotate_" + str(args.rotate) + ".jpg"
   if args.show:
      ia.imshow(image_aug)
   else:
      imageio.imwrite(outputfilename, image_aug)
      print( outputfilename )

def tiling():
   print("Calling tiling method with arg {}".format(args.tiling))
   image = imageio.imread(args.image)
   # Get the y and x dimensions of the image
   image_dimension_y, image_dimension_x, rgb = image.shape
   # Make sure that the new dimensions are multiple of the given tiling size
   next_multiple_of_tiling_dimension_x = int( np.ceil(float(image_dimension_x) / args.tiling) * args.tiling )
   next_multiple_of_tiling_dimension_y = int( np.ceil(float(image_dimension_y) / args.tiling) * args.tiling )
   # Once we have the new dimensions, resize the image
   image = ia.imresize_single_image(image, (next_multiple_of_tiling_dimension_y, next_multiple_of_tiling_dimension_x))
   print( "Original dim x = {}, original dim y = {}, multiple of tiling dim x = {}, multiple of tiling dim y = {}".format( image_dimension_x, image_dimension_y, next_multiple_of_tiling_dimension_x, next_multiple_of_tiling_dimension_y ) )

   count = 1
   for i in range(int(next_multiple_of_tiling_dimension_x / float(args.tiling))):
      for j in range(int(next_multiple_of_tiling_dimension_y / float(args.tiling))):
         # Calculate the position and size of the bounding box (tiling), each time shifting vertically (inner loop) and then horizontally (outer loop)
         bbs = BoundingBoxesOnImage([BoundingBox(x1=(i)*(args.tiling), x2=(i+1)*(args.tiling), y1=(j)*(args.tiling), y2=(j+1)*args.tiling)], shape=(next_multiple_of_tiling_dimension_y, next_multiple_of_tiling_dimension_x) )
         if args.show:
            ia.imshow(bbs.draw_on_image(image, size=2))
         # Extract the calculated bounding box (tiling) from the original input image
         image_aug = bbs.bounding_boxes[0].extend(all_sides=0).extract_from_image(image)
         # Resize the extracted tiling to the original dimension of the input image
         image_aug = ia.imresize_single_image(image_aug, (image_dimension_y, image_dimension_x))
         if args.show:
            ia.imshow(image_aug)
         else:
            outputfilename = args.image[:-4] + "-tiling_" + str(args.tiling) + "x" + str(args.tiling) + "-" + str(count) + ".jpg"
            imageio.imwrite(outputfilename, image_aug)
            print( outputfilename )
         count += 1

def mirror():
   print("Calling mirror method with arg {}".format(args.mirror))
   if not args.mirror == "horizontally" and not args.mirror == "vertically":
      print("Error: the only valid arguments for the mirror method are 'horizontally' or 'vertically'")
      exit(1)

   image = imageio.imread(args.image)
   
   if args.mirror == "horizontally":
      horizontally = iaa.Fliplr(1.0)
      image_aug = horizontally(image=image)
   else:
      vertically = iaa.Flipud(1.0)
      image_aug = vertically(image=image)

   outputfilename = args.image[:-4] + "-mirror_" + str(args.mirror) + ".jpg"
   if args.show:
      ia.imshow(image_aug)
   else:
      imageio.imwrite(outputfilename, image_aug)
      print( outputfilename )

def zoom():
   print("Calling zoom method with arg {}".format(args.zoom))
   image = imageio.imread(args.image)

   zoom = iaa.Affine(scale=(args.zoom))
   image_aug = zoom(image=image)

   outputfilename = args.image[:-4] + "-zoom_" + str(args.zoom) + ".jpg"
   if args.show:
      ia.imshow(image_aug)
   else:
      imageio.imwrite(outputfilename, image_aug)
      print( outputfilename )

def resize():
   print("Calling resize method with arg {}".format(args.resize))

   image = imageio.imread(args.image)

   image_aug = ia.imresize_single_image(image, (args.resize, args.resize))

   outputfilename = args.image[:-4] + "-resize_" + str(args.resize) + "x" + str(args.resize) + ".jpg"
   if args.show:
      ia.imshow(image_aug)
   else:
      imageio.imwrite(outputfilename, image_aug)
      print( outputfilename )

def main(arguments):
   print( "Main: processing image {}".format( args.image ) )
   if args.rotate is not None:
      rotate()
   elif args.tiling is not None:
      tiling()
   elif args.mirror is not None:
      mirror()
   elif args.zoom is not None:
      zoom()
   elif args.resize is not None:
      resize()
   else:
      print("Error: at least one method has to be defined: rotate (-r), tiling (-t), mirror (-m) or zoom (-z)")
      exit()


if __name__ == '__main__':

   parser = argparse.ArgumentParser(add_help=False)
   parser.add_argument('-h','--help', action="help", help="Help")

   parser.add_argument("-r", "--rotate", required=False, type=float, default=None, help="Rotate the image at a given angle")
   parser.add_argument("-t", "--tiling", required=False, type=int, default=None, help="Extract tilings of the given dimension")
   parser.add_argument("-m", "--mirror", required=False, type=str, default=None, help="Mirror the image horizontally or vertically")
   parser.add_argument("-z", "--zoom", required=False, type=float, default=None, help="Zoom the image by a given zoom level")
   parser.add_argument("-re", "--resize", required=False, type=int, default=None, help="Resize the image to the given dimension")
   parser.add_argument("-i", "--image", required=True, help="Input JPEG image")
   parser.add_argument("-s", "--show", required=False, action='store_true', default=False, help="Show the transformed image [default=false]")

   args = parser.parse_args()

   sys.exit(main(sys.argv[1:]))
