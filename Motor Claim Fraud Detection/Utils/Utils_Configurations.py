import ast
from configparser import ConfigParser


class Configuration:
    def __init__(self, config_file_path=""):
        self.config_ = ConfigParser()
        if len(config_file_path) > 0:
            self.config_.read(config_file_path)

    def read_configuration_options(self, section_name, option_name, data_type=None):
        val = None
        if self.config_.has_option(section_name, option_name):
            if data_type == 'bool':
                val = self.config_.getboolean(section_name, option_name)
            elif data_type == 'int':
                val = self.config_.getint(section_name, option_name)
            elif data_type == 'float':
                val = self.config_.getfloat(section_name, option_name)
            elif data_type == 'list':
                val = ast.literal_eval(self.config_.get(section_name, option_name))
            elif data_type == 'dict':
                val = ast.literal_eval(self.config_.get(section_name, option_name))
            else:
                val = self.config_.get(section_name, option_name)
        return val


if __name__ == '__main__':
    pass
