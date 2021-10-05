# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import random
import matplotlib.pyplot as plt

from shrike.compliant_logging import enable_compliant_logging
from shrike.compliant_logging.constants import DataCategory


def run(args):
    n = args.list_length
    list1 = [random.randint(0, 100) for i in range(n)]
    list2 = [random.randint(0, 100) for i in range(n)]

    log = logging.getLogger(__name__)

    log.info(
        "Start metric logging in azure ml workspace portal",
        category=DataCategory.PUBLIC,
    )

    # log list
    log.metric_list(name="list1", value=list1, category=DataCategory.PUBLIC)

    # log table
    log.metric_table(
        name="Lists",
        value={"list1": list1, "list2": list2},
        category=DataCategory.PUBLIC,
    )

    # log scalar value
    log.metric(name="sum1", value=sum(list1), category=DataCategory.PUBLIC)
    log.metric(name="sum2", value=sum(list2), category=DataCategory.PUBLIC)

    # log image
    plt.plot(list1, list2)
    log.metric_image(name="Sample plot", plot=plt, category=DataCategory.PUBLIC)

    # log row
    for i in range(n):
        log.metric_row(
            name="pairwise-sum",
            description="",
            category=DataCategory.PUBLIC,
            pairwise_sum=list1[i] + list2[i],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="SystemLog:")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument(
        "--list_length",
        required=False,
        default=5,
        type=int,
        help="length of test list",
    )
    args = parser.parse_args()

    enable_compliant_logging(
        args.prefix,
        level=args.log_level,
        format="%(prefix)s%(levelname)s:%(name)s:%(message)s",
        use_aml_metrics=True,
    )

    run(args)
