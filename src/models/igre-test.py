import yaml
from jsonschema import validate
import scipy.io
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from src.tftools.optimizer_builder import build_optimizer
from src.models import igre
import os

ROOT_DIR = os.path.abspath(os.curdir)

config_schema = {
    "additionalProperties": True,
    "image": {
        "type": "string"
    },
    "matfile": {
        "required": False
    },
    "crop": {
        "type": "object",
        "properties": {
            "left_top": {
                "type": "object",
                "properties": {
                    "x": "nonNegativeInteger",
                    "y": "nonNegativeInteger"
                },
            },
            "size": {
                "type": "object",
                "properties": {
                    "width": "nonNegativeInteger",
                    "height": "nonNegativeInteger",
                }
            }
        }
    },
    "layers": {
        "type": "array",
        "items": {
            "type": "nonNegativeInteger"
        }
    },
    "input_dimensions": {
        "type": "object",
        "properties": {
            "min": "nonNegativeInteger",
            "max": "nonNegativeInteger"
        }
    },
    "output_dimensions": {
        "type": "object",
        "properties": {
            "min": "nonNegativeInteger",
            "max": "nonNegativeInteger"
        }
    },
    "expected_max_shift_px": "nonNegativeInteger",
    "train": {
        "type": "object",
        "required": True,
        "properties": {
            "batch_size": "nonNegativeInteger",
            "epochs": "nonNegativeInteger",
            "use_gpu": "boolean",
            "optimizer": {
                "required": True,
                "type": "object",
                "properties": {
                    "family": {
                        "required": True,
                        "type": "string"
                    },
                    "learning_rate": {
                        "required": True,
                        "type": "number"
                    },
                    "beta1": "number",
                    "beta2": "number"
                }
            }
        }
    }
}


def check_config(config):
    try:
        validate(config, config_schema)
        print("Config integrity: " + colored("OK", "green"))
    except Exception as e:
        print("Config integrity: " + colored("FAILED", "red"))
        print(e)
        exit(1)


def data_crop(config, dataset):
    if "crop" in config:
        print("Data crop ... " + colored("YES", "green") + ":")
        print("\t["
              + str(config["crop"]["left_top"]["x"]) + ":"
              + str(config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]) + ", "
              + str(config["crop"]["left_top"]["y"]) + ":"
              + str(config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]) + ", :]")
        dataset = dataset[
                  config["crop"]["left_top"]["x"]: (config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]),
                  config["crop"]["left_top"]["y"]: (config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]),
                  :]
    else:
        print("Data crop: " + colored("NO", "red"))

    return dataset


def igre_test(config, shift, output):
    """
    Information gain and registration test works with registered inputs. For testing registration layer, input pixels
    are shifted by shift[0] in x axis and by shift[1] in y axis.
    :param config: configuration of IGRE run
    :param shift: tuple containing shift in x and in y axis of input dimensions
    :param output: output file for measured data
    :return: registration layer weights (i.e. computed shift)
    """

    # Config load and integrity check
    config = yaml.load(open(os.path.join(ROOT_DIR, config), 'r'), Loader=yaml.FullLoader)
    check_config(config)

    print("\nWelcome to " + colored("IGRE-test", "green") + " run with file: " + colored(config["matfile"], "green") +
          " expected shift: " + colored(shift, "green") + "\n")

    dataset = np.float64(scipy.io.loadmat(config["matfile"])['data'])
    print("Data stats (before normalization): min = " + str(np.min(dataset)) + " max = " + str(np.max(dataset)))
    # data normalization - ranged
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

    print("Dataset shape: " + str(dataset.shape))
    dataset = data_crop(config, dataset)

    print(colored("Input", "green") + " dimensions: " + colored(("[" +
                                                                 str(config["input_dimensions"]["min"]) + "-" +
                                                                 str(config["input_dimensions"]["max"]) + "]"),
                                                                "green"))
    visible = dataset[:, :, config["input_dimensions"]["min"]: config["input_dimensions"]["max"] + 1]
    plt.imshow(visible[:,:,0], cmap='gray')
    plt.show()
    indexes = np.indices(visible.shape[:-1]).reshape((len(visible.shape[:-1]), -1)).transpose().astype(np.float32)

    print("\tInputs shape: " + str(visible.shape))
    print(colored("Output", "green") +
          " dimensions: " + colored(("[" +
          str(config["output_dimensions"]["min"]) + "-" +
          str(config["output_dimensions"]["max"]) + "]"),
         "green"))
    outputs = dataset[:, :, config["output_dimensions"]["min"]: config["output_dimensions"]["max"] + 1]
    print("\tOutput shape: " + str(outputs.shape))

    print("\nCalling " + colored("IGRE\n", "green") + "...")

    bias, bias_history = igre.run(indexes + shift,
                                  outputs,
                                  visible=visible,
                                  optimizer=build_optimizer(config["train"]["optimizer"]),
                                  layers=config["layers"],
                                  batch_size=config["train"]["batch_size"],
                                  epochs=config["train"]["epochs"]
                                  )

    if output is not None:
        with open(output, 'w') as ofile:
            config["bias"] = {
                "x": float(bias[0][0]),
                "y": float(bias[0][1])
            }
            config["bias_history"] = {
                "x": [float(hist[0][0]) for hist in bias_history],
                "y": [float(hist[0][1]) for hist in bias_history]
            }
            ofile.write(yaml.dump(config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        # WARNING: keep this up to date to relative path of both files
        default="data/external/metacentrum/01-registration-experiment/config.yaml",
        help="Config file for IGRE. For more see example file.",
    )
    parser.add_argument(
        "-x",
        "--x-shift",
        type=float,
        default=0,
        help="x-Shift of the input data",
    )
    parser.add_argument(
        "-y",
        "--y-shift",
        type=float,
        default=0,
        help="y-Shift of the input data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="yaml output file, where collected data will be placed",
    )
    args = parser.parse_args()
    igre_test(args.config, (args.x_shift, args.y_shift), args.output)
