import torch

from torch import nn
from torchvision import models as tm
from torch.nn import functional as tnf


from collections import OrderedDict



class PSPNet( nn.Module ):

    def __init__(self, output_channels = 3 ):
        super().__init__()

        self.resnet = tm.resnet34( pretrained = True )

        self.resnet = nn.Sequential( *list(self.resnet.children() )[:-2] )

        self.AdaptiveAvgPooling1Conv = nn.Sequential(  OrderedDict( [ ( "AdapAvg1",nn.AdaptiveAvgPool2d( output_size = 1) ), 
                                ( "AdapAvg1Conv", nn.Conv2d(in_channels = 512, 
                                out_channels = 128, 
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False ) 
                                ), 
                                ( "BatchNorm1", nn.BatchNorm2d( num_features = 128 ))
                                ] ) 
                            )

        self.AdaptiveAvgPooling2Conv = nn.Sequential(  OrderedDict( [ ( "AdapAvg2", nn.AdaptiveAvgPool2d( output_size = 2 ) ), 
                                ( "AdapAvg1Conv", nn.Conv2d(in_channels = 512, 
                                out_channels = 128, 
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False ) 
                                ),

                                ( "BatchNorm2", nn.BatchNorm2d( num_features = 128 ))
                                 ] ) 
                            )

        self.AdaptiveAvgPooling3Conv = nn.Sequential(  OrderedDict( [ ( "AdapAvg3", nn.AdaptiveAvgPool2d( output_size = 3 ) ), 
                                ( "AdapAvg2Conv", nn.Conv2d(in_channels = 512, 
                                out_channels = 128, 
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False ) 
                                ),
                                ( "BatchNorm3", nn.BatchNorm2d( num_features = 128 ))
                                 ] ) 
                            )

        self.AdaptiveAvgPooling6Conv = nn.Sequential(  OrderedDict( [ ( "AdapAvg6", nn.AdaptiveAvgPool2d( output_size = 6 ) ), 
                                ( "AdapAvg3Conv", nn.Conv2d(in_channels = 512, 
                                out_channels = 128, 
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False ) 
                                ),
                                ( "BatchNorm6", nn.BatchNorm2d( num_features = 128 ))
                                 ] ) 
                            )

        

        self.conv1 = nn.Sequential ( OrderedDict([ 
            ( "ConvAfterConcatenation", nn.Conv2d( in_channels = 1024 , 
                                out_channels = 128, 
                                kernel_size = 1, 
                                stride = 1,
                                padding = 0,
                                bias = False,
                                ) ),
            ("BatchNormBeforeOutput", nn.BatchNorm2d( num_features = 128 ))
                    ])
        )

        self.conv2 = nn.Conv2d( in_channels = 128,
                                out_channels = 3,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False
        )


        self.softmax = nn.Softmax( dim = 1 )


    def forward( self, input_tensor ):
        
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        output_tensor = self.resnet( input_tensor )


        output_h, output_w = output_tensor.shape[2], output_tensor.shape[3]

        up1 = self.AdaptiveAvgPooling1Conv( output_tensor )
        up2 = self.AdaptiveAvgPooling2Conv( output_tensor )
        up3 = self.AdaptiveAvgPooling3Conv( output_tensor )
        up6 = self.AdaptiveAvgPooling6Conv( output_tensor )

        up1 = tnf.upsample_bilinear( input = up1, size = ( output_h, output_w ) )
        up2 = tnf.upsample_bilinear( input = up2, size = ( output_h, output_w ) )
        up3 = tnf.upsample_bilinear( input = up3, size = ( output_h, output_w ) )
        up6 = tnf.upsample_bilinear( input = up6, size = ( output_h, output_w ) )


        output_tensor = torch.cat( [ output_tensor, up1, up2, up3, up6 ], dim = 1 )

        output_tensor = self.conv1( output_tensor )

        output_tensor = tnf.upsample_bilinear( input = output_tensor, size = ( input_h, input_w ) )

        output_tensor = self.conv2( output_tensor )

        output_tensor = self.softmax( output_tensor )

        

        return output_tensor