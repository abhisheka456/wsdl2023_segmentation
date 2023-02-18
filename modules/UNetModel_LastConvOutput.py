from simplejson import OrderedDict
import torch

from torch import nn
from torchvision.transforms import functional as ttf


from collections import OrderedDict


class DoubleConvWithRelu( nn.Module ):


    def __init__(self, level : str, input_channels : int , output_channels : int ):
        super().__init__()

        self.contracting_path_unit = nn.Sequential(
            OrderedDict(

                [
                       ( "Level_" + level + "_conv1",  nn.Conv2d( in_channels = input_channels,
                       out_channels = output_channels, 
                       kernel_size = 3,
                       stride = 1,
                       padding = 1 ) ),

                       
                       ( "Level_" + level + "_relu1", nn.ReLU(inplace = True ) ),

                       ( "Level_" + level + "_conv2", nn.Conv2d( in_channels = output_channels,
                       out_channels = output_channels,
                       kernel_size = 3,
                       stride = 1,
                       padding = 1 ) ),

                       ( "Level_" + level + "_relue2", nn.ReLU( inplace = True ) ), 
                        ( "Level_" + level + "_Batch_Normalization", nn.BatchNorm2d(
                            num_features= output_channels ) )


                ]
            )
        )

        self.maxpool = nn.MaxPool2d( kernel_size = 2, stride = 2 )



    def forward( self, input_tensor ):

        output = self.contracting_path_unit( input_tensor )
        maxpool_output = self.maxpool( output )


        return output, maxpool_output



class UpConv( nn.Module ):

    def __init__(self, level : str, input_channels : int, output_channels : int ):
        super().__init__()

        self.upsample = nn.Upsample( scale_factor = 2 )

        self.conv2x2 =\
        nn.Sequential( OrderedDict([
            ( "Level_"+ level + "_conv2x2", 
            nn.Conv2d( in_channels = input_channels,
            out_channels = output_channels,
            kernel_size = 2, 
            stride = 1,
            padding = ( 1, 1 ) ) 
            ) ]) )

        self.expansive_path_unit = nn.Sequential( OrderedDict( [ 
            ( "Level_"+ level +"_conv1_for_expansive", nn.Conv2d( in_channels = input_channels,
            out_channels = output_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,) ),
            ("Level_" + level + "_relu1_for_expansive", nn.ReLU( inplace = True ) ), 
            ("Level_" + level +"_conv2_for_expansive", nn.Conv2d( in_channels = output_channels,
            out_channels = output_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1) ),
            ("Level_" + level + "_relue2_for_expansive", nn.ReLU( inplace = True ) ),
            ("Level_" + level + "_Batch_Normalization_of_expansive", nn.BatchNorm2d(
                num_features = output_channels 
            )) 

         ] )
        )



    def forward( self, input_tensor, contracting_path_tensor ):

        output_tensor = self.upsample( input_tensor )

        output_tensor = self.conv2x2(  output_tensor )

        output_tensor = output_tensor[ :, : , 1: , 1: ]

        # print( input_tensor.shape, output_tensor.shape, contracting_path_tensor.shape )

        output_tensor = torch.cat( [ output_tensor, contracting_path_tensor ], dim = 1 )

        # print( output_tensor.shape )

        output_tensor = self.expansive_path_unit( output_tensor )

        return output_tensor 



class UNet( nn.Module ):

    def __init__(self):
        super().__init__()

        self.contracting_level1 = DoubleConvWithRelu( "1", input_channels = 3, 
        output_channels = 64 ) 
        self.contracting_level2 = DoubleConvWithRelu( "2", input_channels = 64,
        output_channels = 128 )
        self.contracting_level3 = DoubleConvWithRelu( "3", input_channels = 128,
        output_channels = 256 )
        self.contracting_level4 = DoubleConvWithRelu( "4", input_channels = 256,
        output_channels = 512)

        self.backbone = DoubleConvWithRelu("backbone", input_channels = 512,
        output_channels = 1024 )

        self.expansive_level4 = UpConv( "4", input_channels = 1024,
        output_channels = 512 )
        self.expansive_level3 = UpConv( "3", input_channels = 512,
        output_channels = 256 )
        self.expansive_level2 = UpConv( "2", input_channels = 256,
        output_channels = 128 )
        self.expansive_level1 = UpConv( "1", input_channels = 128,
        output_channels = 64 )


        self.segmentation_layer = nn.Conv2d( in_channels = 64,
        out_channels = 3, 
        kernel_size = 1,
        stride = 1,
        padding = 0 )

        self.softmax = nn.Softmax( dim = 1 )






    def forward( self, input_tensor ):

        contractive_output1, contractive_maxpool1 = self.contracting_level1( input_tensor )
        contractive_output2, contractive_maxpool2 = self.contracting_level2( contractive_maxpool1 )
        contractive_output3, contractive_maxpool3 = self.contracting_level3( contractive_maxpool2 )
        contractive_output4, contractive_maxpool4 = self.contracting_level4( contractive_maxpool3 )


        backbone_output, _ = self.backbone( contractive_maxpool4 )
        
        expansive_output4 = self.expansive_level4( backbone_output, contractive_output4 )
        expansive_output3 = self.expansive_level3( expansive_output4, contractive_output3 )
        expansive_output2 = self.expansive_level2( expansive_output3, contractive_output2 )
        expansive_output1 = self.expansive_level1( expansive_output2, contractive_output1 )
        
        segmentation_output = self.segmentation_layer( expansive_output1 )

        segmentation_output = self.softmax( segmentation_output )

        return segmentation_output, expansive_output1 

