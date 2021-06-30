# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
import math
import os
from os import path

from azureml.core.run import Run

CHECKPOINT_FILE_NAME = "model_checkpoint.txt"


def get_logger():
    try:
        return Run.get_context()
    except Exception:
        return LocalLogger()


class LocalLogger:
    def log(self, key, value):
        print("AML-Log:", key, value)


def evaluate(num_epochs, delay_seconds, x1, x2, checkpoint_location):
    run_logger = get_logger()
    start_epoch = 0
    if checkpoint_location:
        try:
            checkpoint_file_path = os.path.join(
                checkpoint_location, CHECKPOINT_FILE_NAME
            )
            with open(checkpoint_file_path) as f:
                start_epoch = int(f.readline())
            print(
                "Found a checkpoint. Starting training from epoch: {}".format(
                    start_epoch
                )
            )
        except Exception as ex:
            print(
                "Exception occurred while trying to read checkpoint file. "
                "Starting training from epoch 0: {}".format(ex)
            )

    for ep in range(start_epoch, num_epochs):
        # x1 = 0, x2 = 0 maximize the result.
        result = math.sqrt(ep * 1000) / (1 + math.pow(x1, 2) + math.pow(x2, 2))
        result /= math.sqrt(num_epochs * 1000)  # Results between 0 and 1
        run_logger.log("result", float(result))
        time.sleep(delay_seconds)

        # Write checkpoints
        output_checkpoint_dir = "./outputs/"
        if not os.path.exists(output_checkpoint_dir):
            os.makedirs(output_checkpoint_dir)
        output_checkpoint_file_path = os.path.join(
            output_checkpoint_dir, CHECKPOINT_FILE_NAME
        )
        with open(output_checkpoint_file_path, "w") as f:
            f.write(str(ep) + "\n")


def main():
    # Get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, default=20
    )  # Number of metrics per child run
    parser.add_argument(
        "--delay_seconds", type=float, default=20
    )  # Delay after logging a metric
    parser.add_argument("--x1", type=float, default=0, help="input x1")
    parser.add_argument("--x2", type=float, default=0, help="input x2")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="location of the model or checkpoint files from where to resume the training",
    )

    args = parser.parse_args()

    # log parameters
    run_logger = get_logger()
    run_logger.log("args", args)

    previous_checkpoint_location = args.resume_from

    evaluate(
        args.num_epochs,
        args.delay_seconds,
        args.x1,
        args.x2,
        previous_checkpoint_location,
    )


if __name__ == "__main__":
    main()
