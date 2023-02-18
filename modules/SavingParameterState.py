import os
import torch
from torch import nn

def save_model_parameters( model : nn.Module,
 optimizer : torch.optim.Optimizer, 
 name_of_the_model : str ):

    model_parameter_state = dict(
        {

            "BestModel" : model.state_dict( ),
            "Parameters" : optimizer.state_dict( )
        }
    )


    destination_path =  os.path.join( os.path.dirname( __file__ ), "BestModels", name_of_the_model + ".pth" )

    torch.save( obj = model_parameter_state, f = destination_path )

    print( "The best model is saved for {0} at location {1}.".format( name_of_the_model, destination_path ) )



def load_model_parameter( model : nn.Module,
                        optimizer : torch.optim.Optimizer,
                        name_of_the_model : str 
):

    destination_path =  os.path.join( os.path.dirname( __file__ ), "BestModels", name_of_the_model + ".pth" )

    model_parameter_state = torch.load( f = destination_path )

    model.load_state_dict( state_dict = model_parameter_state[ "BestModel" ] )
    
    optimizer.load_state_dict( state_dict = model_parameter_state[ "Parameters" ] )

