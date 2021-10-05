# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Simple script with examples of how to use "with statements" to capture
information about failed module imports.
"""

from shrike.compliant_logging.exceptions import PrefixStackTrace

# Output will be:
#
# SystemLog: Traceback (most recent call last):
# SystemLog:   File ".\docs\logging\with-statement.py", line 11, in <module>
# SystemLog:     import my_custom_library  # noqa: F401
# SystemLog: ModuleNotFoundError: **Exception message scrubbed**

with PrefixStackTrace():
    # Import statement which could raise an exception containing sensitive
    # data.
    import my_custom_library  # noqa: F401

# Output will be:
#
# SystemLog: Traceback (most recent call last):
# SystemLog:   File ".\docs\logging\with-statement.py", line 22, in <module>
# SystemLog:     import another_custom_library  # noqa: F401
# SystemLog: ModuleNotFoundError: No module named 'another_custom_library'
with PrefixStackTrace(keep_message=True):
    # Import statement which will never raise an exception containing sensitive
    # data.
    import another_custom_library  # noqa: F401
