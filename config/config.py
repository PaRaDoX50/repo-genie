from os import environ, getenv, makedirs
from platform import system
from os.path import expanduser, join
from subprocess import run

import toml
from dynaconf import Dynaconf

VERSION = "1.5.0"
BASE_PATH = "/Users/sumedhzope/repo-genie"
CONFIG_FILENAME = "config.toml"
CONFIG_PATH = join(BASE_PATH, "config")
STORAGE_PATH = join(BASE_PATH, "storage")
INDEX_CACHE_FILENAME = "index_cache.sqlite"
INDEX_CACHE_PATH = join(BASE_PATH, "index")

def get_file_path(path, filename):
    expanded_path = expanduser(path)
    makedirs(expanded_path, exist_ok=True)
    return join(expanded_path, filename)

def load_config(skip_environment_vars=False):
    config_object = Dynaconf(
        settings_files=[get_file_path(CONFIG_PATH, CONFIG_FILENAME)]
    )
    config_dict = config_object.as_dict()

    # Set LiteLLM API keys only if not already set in environment
    for key, value in config_dict["CONF"]["LITELLM_API_KEYS"].items():
        if key.endswith("_API_KEY") and value and key not in environ:
            environ[key] = value

    return config_dict
