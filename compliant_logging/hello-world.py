# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Simplest example of how to use the prefix_stack_trace decorator.
"""

from shrike.compliant_logging.exceptions import prefix_stack_trace


# Output will be:
# SystemLog: Traceback (most recent call last):
# SystemLog:   File ".\hello-world.py", line 11, in main
# SystemLog:     print(1 / 0)
# SystemLog: ZeroDivisionError: **Exception message scrubbed**
@prefix_stack_trace()
def main():
    print("Hello, world!")
    print(1 / 0)


if __name__ == "__main__":
    main()
