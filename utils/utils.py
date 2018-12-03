import json


def save_config(filepath, config):
    with open(filepath, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)


def load_config(filepath):
    with open(filepath, "r") as infile:
        config = json.load(infile)

    return config
