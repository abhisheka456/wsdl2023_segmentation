from sqlalchemy import true
import torch
from torch.utils.data import Dataset
from . import DataDirectory as dd

import os
import skimage.io as skio
import skimage.transform as sktr
import skimage.util as skut
import numpy as np


class TrainDataGenerator( Dataset ):

    def __init__(self, species_name : str ) -> None:
        super().__init__()

        self.train_image_list, _ = dd.getTrainValFileNameFromTxt( species_name = species_name )

        self.train_image_object_list = list()
        self.train_mask_object_list = list() 

        for i_filename in self.train_image_list:

            image_file_path = os.path.join( dd.getImageDirectory(), i_filename + ".jpg"  ) 
            mask_file_path = os.path.join( dd.getMaskDirectory(), i_filename + ".png" )

            if os.path.exists( image_file_path ) and os.path.exists( mask_file_path ):

                image_object = dd.getObjectOfAnImage( image_file_path )
                mask_object = dd.getObjectOfAnImage( mask_file_path, as_gray= true )


                mask_object = skut.img_as_ubyte( image = mask_object )
                

                mask_object = mask_object - 1

                
                image_object = sktr.resize( image = image_object, output_shape = ( 256, 256 ),
                preserve_range = true )
                mask_object = sktr.resize( order = 0,  image = mask_object, output_shape = [ 256, 256 ],
                preserve_range = true )


                self.train_image_object_list.append( image_object )
                self.train_mask_object_list.append( mask_object )


        



    def __len__( self ):

        return len( self.train_image_list )

    def __getitem__(self, index):

        image_object = self.train_image_object_list[ index ]
        mask_object = self.train_mask_object_list[ index ]

        if np.random.random() > 0.5:

            angle_to_rotate = np.random.randint( low = 0,
            high = 360)

            image_object = sktr.rotate( image = image_object, angle = angle_to_rotate ) 
            mask_object = sktr.rotate( image = mask_object, angle = angle_to_rotate,
             preserve_range = true )


            mask_object =   mask_object.astype( dtype = np.uint8 )

            
        image_object, mask_object = dd.getObjectInTensor( [ image_object, mask_object ] )



        return  image_object, mask_object



class ValDataGenerator( Dataset ):

    def __init__( self, species_name ) -> None:
        super().__init__()

        _ , self.val_image_list = dd.getTrainValFileNameFromTxt( species_name = species_name )


        self.val_image_object_list = list()
        self.val_mask_object_list = list()

        for i_filename in self.val_image_list:

            image_file_path = os.path.join( dd.getImageDirectory(), i_filename + ".jpg"  ) 
            mask_file_path = os.path.join( dd.getMaskDirectory(), i_filename + ".png" )

            if os.path.exists( image_file_path ) and os.path.exists( mask_file_path ):

                image_object = dd.getObjectOfAnImage( image_file_path )
                mask_object = dd.getObjectOfAnImage( mask_file_path )

                mask_object = skut.img_as_uint( image = mask_object )
                
                image_object = sktr.resize( image = image_object, output_shape = ( 256, 256 ))
                mask_object = sktr.resize( image = mask_object, output_shape = [ 256, 256 ] )

                self.val_image_object_list.append( image_object )
                self.val_mask_object_list.append( mask_object )




    def __len__( self ):

        return len( self.val_image_list )


    def __getitem__(self, index) :
        
        image_object = self.val_image_object_list[ index ]
        mask_object = self.val_mask_object_list[ index ]

        image_object, mask_object = dd.getObjectInTensor( [ image_object, mask_object ] )

        return image_object, mask_object.to( dtype = torch.long ) 



class TestDataGenerator( Dataset ):

    """

    A Test Data Generator for Oxford IIIT dataset
    
    """

    def __init__(self, species_name : str ) :
        super().__init__()

        self.test_image_list = dd.getTestFileNameFromTxt( species_name = species_name )

        self.test_image_object_list = list()
        self.test_mask_object_list = list()


        for i_filename in self.test_image_list:

            image_file_path = os.path.join( dd.getImageDirectory(), i_filename + ".jpg"  ) 
            mask_file_path = os.path.join( dd.getMaskDirectory(), i_filename + ".png" )


            if os.path.exists( image_file_path ) and os.path.exists( mask_file_path ):

                image_object = dd.getObjectOfAnImage( image_file_path )
                mask_object = dd.getObjectOfAnImage( mask_file_path )
                
                mask_object = skut.img_as_uint( image = mask_object )

                image_object = sktr.resize( image = image_object, output_shape = ( 256, 256 ))
                mask_object = sktr.resize( image = mask_object, output_shape = [ 256, 256 ] )

                self.test_image_object_list.append( image_object )
                self.test_mask_object_list.append( mask_object )
                



    def __len__(self):

        return len( self.test_image_list )


    def __getitem__(self, index) :

        image_object = self.test_image_object_list[ index ]
        mask_object = self.test_mask_object_list[ index ]

        image_object, mask_object = dd.getObjectInTensor( [ image_object, mask_object ] )

        return image_object, mask_object.to( dtype = torch.long)