import os
from matplotlib import image

import numpy as np
from numpy import dtype


import skimage.io as skio
import skimage.util as skut

import torch

from torchvision.transforms import functional as ttf



def getRootDataDirectory():

    """
    The root directory location.

    """

    path_to_data_directory = os.path.abspath( os.path.join( os.path.dirname (__file__), "..",
     "..", "..", "WIP", "Data", "Dataset", "Oxford_IIIT" ) )
    # print(  path_to_data_directory )

    # print( os.path.exists( path_to_data_directory ) )

    return path_to_data_directory

def getImageDirectory():

    """
    This function returns images directory path as string.

    """

    path_to_image_directory = os.path.join( getRootDataDirectory(), 
    "images"
    )


    if os.path.exists( path_to_image_directory ):

        # print( path_to_image_directory )
        return path_to_image_directory



def getMaskDirectory():

    """
    
    This function returns masks directory path as string.
    
    """

    path_to_mask_directory = os.path.join( getRootDataDirectory(),
    "annotations", "trimaps"
    )


    if os.path.exists( path_to_mask_directory ):

        return path_to_mask_directory

def getImagesOfASpecies( species_name : str ):

    """

    This function returns a set of file paths for a given species.

    ### Parameters
    ---

    :param species_name: The name of the particular species.

    """

    path_to_images = list()

    for filename in os.listdir( getImageDirectory() ):

        if filename.startswith( species_name ):

            path_to_images.append( os.path.join( getImageDirectory(), filename ) )


    path_to_images.sort()

    return path_to_images



def getMasksOfASpecies( species_name : str ):

    path_to_masks = list()


    for filename in os.listdir( getMaskDirectory() ):

        if filename.startswith( species_name ):

            path_to_masks.append( os.path.join( getMaskDirectory(), filename )
                )


    path_to_masks.sort()

    return path_to_masks



def getObjectOfAnImage( path_to_a_mask : str, as_gray = False ):
    """
    
    ### Parameters:

    :param path_to_a_mask:  A path in string.


    """

    pic_object = skio.imread( path_to_a_mask, as_gray = as_gray )

    if not as_gray :
        pic_object = skut.img_as_float32( image = pic_object )


    return pic_object



def getTrainValFileNameFromTxt( species_name : str, train_size = 0.7 ):


    trainval_filename_list = list()

    with open( os.path.join( getRootDataDirectory(), "annotations", "trainval.txt" ), "r" ) as textfile:

        linesread = textfile.readlines()

        for i_line in linesread:

            if i_line.startswith( species_name ):

                trainval_filename_list.append(  i_line.split()[0]  )

    train_len = int ( len( trainval_filename_list ) * train_size )

    train_filename_list = trainval_filename_list[ : train_len ]
    val_filename_list = trainval_filename_list[ train_len : ]

    return train_filename_list, val_filename_list


def getTestFileNameFromTxt( species_name : str ):

    test_filename_list = list()

    with open( os.path.join( getRootDataDirectory(), "annotations", "test.txt" ) ) as textfile:

        linesread = textfile.readlines()

        for i_line in linesread:

            if i_line.startswith( species_name ):

                test_filename_list.append(  i_line.split()[0] )

    return test_filename_list


def getObjectInTensor( list_of_objects : list ):

    """
    This function returns input object list into tensor

    """

    image_object, mask_object = list_of_objects

    tensor_list =[ ttf.to_tensor( image_object ),
                        torch.tensor( data = mask_object, dtype = torch.long )
     ] 

    return tensor_list