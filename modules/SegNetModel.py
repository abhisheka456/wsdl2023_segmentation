from re import I
import torch
import numpy as np

from torch import nn

from torchvision.transforms import functional as ttf

from collections import OrderedDict


class SegNet( nn.Module ):

    def __init__(self):
        super().__init__()
################################################################################
#
#
#      Encoder Section
#
#
################################################################################
        self.encoder_level_1 = nn.Sequential(
            OrderedDict(  [  "EncoderLevel1Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 3 if i_index == 0 else 64,
            output_channels = 64   ) ] for i_index in range(2)  
            
              )
        ) 

        self.encoder_level_2 = nn.Sequential(

            OrderedDict(  [  "EncoderLevel2Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 64 if i_index == 0 else 128,
            output_channels = 128   ) ] for i_index in range(2)  
            
              )
        )


        self.encoder_level_3 = nn.Sequential(

            OrderedDict(  [  "EncoderLevel3Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 128 if i_index == 0 else 256,
            output_channels = 256   ) ] for i_index in range(3)  
            
              )
        )

        self.encoder_level_4 = nn.Sequential(

            OrderedDict(  [  "EncoderLevel4Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 256 if i_index == 0 else 512,
            output_channels = 512   ) ] for i_index in range(3)  
            
              )
        )


        self.encoder_level_5 = nn.Sequential(

            OrderedDict(  [  "EncoderLevel5Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 512,
            output_channels = 512   ) ] for i_index in range(3)  
            
              )
        )



        self.maxpool = nn.MaxPool2d( kernel_size = 2,
        stride = 2,
        return_indices = True,
         )

        self.maxunpool = nn.MaxUnpool2d( kernel_size = 2,
        stride = 2 )



        self.decoder_level_5 = nn.Sequential(

            OrderedDict(  [  "DecoderLevel5Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 512,
            output_channels = 512   ) ] for i_index in range(3)  
            
              )
        )

        self.decoder_level_4 = nn.Sequential(

            OrderedDict(  [  "DecoderLevel4Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 512,
            output_channels = 256 if i_index == 2 else 512  ) ] for i_index in range(3)  
            
              )
        )


        self.decoder_level_3 = nn.Sequential(

            OrderedDict(  [  "DecoderLevel3Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 256,
            output_channels = 128 if i_index == 2 else 256  ) ] for i_index in range(3)  
            
              )
        )



        self.decoder_level_2 = nn.Sequential(

            OrderedDict(  [  "DecoderLevel2Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 128,
            output_channels = 64 if i_index == 1 else 128  ) ] for i_index in range(2)  
            
              )
        )


        self.decoder_level_1 = nn.Sequential(

            OrderedDict(  [  "DecoderLevel4Index{0}".format( i_index ) , ConvBatchNorm( level = 1, 
            index = i_index, 
            input_channels = 64,
            output_channels = 3 if i_index == 1 else 64  ) ] for i_index in range(2)  
            
              )
        )

        self.softmax = nn.Softmax( dim = 1 )

        # softmax_i = exp ( x_i ) / for  all i sum( x_i )

    def forward( self, input_tensor ):

        input_size = input_tensor.size()

        output_tensor = self.encoder_level_1( input_tensor )
        output_tensor, en1_max_indices = self.maxpool( output_tensor )

        en1_size = output_tensor.size()

        output_tensor = self.encoder_level_2( output_tensor )
        output_tensor, en2_max_indices = self.maxpool( output_tensor )

        en2_size = output_tensor.size()

        output_tensor = self.encoder_level_3( output_tensor )
        output_tensor, en3_max_indices = self.maxpool( output_tensor )

        en3_size = output_tensor.size()


        output_tensor = self.encoder_level_4( output_tensor )
        output_tensor, en4_max_indices = self.maxpool( output_tensor )

        en4_size = output_tensor.size()

        output_tensor = self.encoder_level_5( output_tensor )
        output_tensor, en5_max_indices = self.maxpool( output_tensor )



        output_tensor = self.maxunpool( input = output_tensor, indices = en5_max_indices,
         output_size = en4_size )
        output_tensor = self.decoder_level_5( output_tensor )

        output_tensor = self.maxunpool( input = output_tensor, indices = en4_max_indices,
        output_size = en3_size )
        output_tensor = self.decoder_level_4( output_tensor )

        output_tensor = self.maxunpool( input = output_tensor, indices = en3_max_indices,
        output_size = en2_size )
        output_tensor = self.decoder_level_3( output_tensor )


        output_tensor = self.maxunpool( input = output_tensor, indices = en2_max_indices,
        output_size = en1_size )
        output_tensor = self.decoder_level_2( output_tensor )


        output_tensor = self.maxunpool( input = output_tensor, indices = en1_max_indices,
        output_size = input_size )
        output_tensor = self.decoder_level_1( output_tensor )


        output_tensor = self.softmax( output_tensor )


        return output_tensor 

class ConvBatchNorm( nn.Module ):

    """
        It is a class for repetative application of Convolution and Batch Normalization. 
    
    
    """

    def __init__(self, level : int, index : int, input_channels : int, output_channels : int ):
        super().__init__()

        self.conv_batch_normalization_unit = nn.Sequential(
            OrderedDict([

                ( "Conv{0}{1}".format( level, index ), nn.Conv2d( in_channels = input_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1, 
                 ) ),
                ("BatchNorm{0}{1}".format( level, index ), nn.BatchNorm2d( num_features = output_channels ) )

            ])

        )


    def forward( self, input_tensor ):

        output_tensor = self.conv_batch_normalization_unit( input_tensor )

        return output_tensor 



