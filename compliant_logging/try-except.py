# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Simple script with examples of how to directly use the function
print_prefixed_stack_trace to capture information about failed module imports.
"""

from shrike.compliant_logging.exceptions import print_prefixed_stack_trace_and_raise

try:
    # Import statement which could raise an exception containing sensitive
    # data.
    import my_custom_library  # noqa: F401
except BaseException as e:
    # Output will be:
    #
    # SystemLog: Traceback (most recent call last):
    # SystemLog:   File ".\try-except.py", line 10, in <module>
    # SystemLog:     import my_custom_library
    # SystemLog: ModuleNotFoundError: **Exception message scrubbed**
    print_prefixed_stack_trace_and_raise(err=e)

try:
    # Import statement which will never raise an exception containing sensitive
    # data.
    import another_custom_library  # noqa: F401
except BaseException as e:
    # Output will be:
    #
    # SystemLog: Traceback (most recent call last):
    # SystemLog:   File ".\try-except.py", line 17, in <module>
    # SystemLog:     import another_custom_library
    # SystemLog: ModuleNotFoundError: No module named 'another_custom_library'
    print_prefixed_stack_trace_and_raise(err=e, keep_message=True)
