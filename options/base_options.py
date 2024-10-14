import argparse
import yaml
import datetime

class BaseOptions():
    """
    This class defines options used both during training and test time. 

    It loads the yaml configuration file. 
    """

    def __init__(self):
        """Reset the class; inidicates that the class has not been initialized yet. """

        self.initialized = False 
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")[2:] # e.g. 240711-161845

    def gather_options(self):
        
        # check if self.initialized 
        assert not self.initialized

        parser = argparse.ArgumentParser(description='Set options for the NeRF-Bioimage scripts')

        # add an argument for parser 
        parser.add_argument('--config_path', help ='specify the path to the YAML configuration file. ')
        args = parser.parse_args()

        self.yaml_dir = args.config_path
                    
        with open(self.yaml_dir, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        recursive_attr(self, yaml_config)
        
        if not hasattr(self, 'load_iter'):
            self.load_iter = 0 
        if not hasattr(self, 'epoch_count'):
            self.epoch_count = 0 
        if not hasattr(self, 'continue_train'):
            self.continue_train = False 
        if not hasattr(self, 'verbose'):
            self.verbose = False 

        setattr(self, 'time', self.time)
        setattr(self, 'name', f"{self.name}_exp_{self.time}") # type: ignore 

        # # modify model-related parser options
        # model_name = self.model
        # model_option_setter = models.get_option_setter(model_name) # initialize model-specific conditions
        # parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # # modify dataset-related parser options
        # dataset_name = self.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name) 
        # parser = dataset_option_setter(parser, self.isTrain)

        return self

@staticmethod
def recursive_attr(class_instance, dict_):
    '''
    If YAML file contains a nested dictionary, this function will iterate to the deepest depth to parse all configs
    '''
    for k, v in dict_.items():
        if isinstance(v, dict):
            new_dict =  dict(zip(v.keys(), v.values()))
            recursive_attr(class_instance, new_dict)
        else:
            setattr(class_instance, k, v)
