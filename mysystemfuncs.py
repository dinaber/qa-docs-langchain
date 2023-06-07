import configparser
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_key_from_config(key):
    config = configparser.ConfigParser()
    config.read(os.path.join(dir_path, 'config.ini'))

    return config['DEFAULT'][key]


def return_num_words(file_path):
    file = open(file_path, "r")
    file_contents = file.read()
    file.close()

    words = file_contents.split()
    return len(words)
