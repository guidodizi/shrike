## Exception Handling

First execute `pip install shrike` to install this library. Then
wrap any methods which may throw an exception with the decorator
`prefix_stack_trace`. Here's a simple example. Your code may explicitly raise
the `Public*` exceptions (`PublicValueError`, `PublicRuntimeError`,
`PublicArgumentError`, `PublicKeyError`, `PublicTypeError`) when you know that
the content of the exception does not contain any private content. The messages
in these exceptions will be preserved, even if `keep_message` is set to `False`.

```python
from shrike.compliant_logging.exceptions import prefix_stack_trace

@prefix_stack_trace()
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
```

## Logging

Call `shrike.compliant_logging.enable_compliant_logging` to set up 
data-category-aware logging. Then continue to use standard Python logging
functionality as before! Add a `category=DataCategory.PUBLIC` argument to have
your log lines prefixed with `SystemLog:`. Here is a full-fledged example:

```python
{!docs/compliant_logging/data-category.py!}
```

## Examples

The simplest use case (wrap your `main` method in a decorator) is:

```python
{!docs/compliant_logging/hello-world.py!}
```

### Prefixing stack trace

Some configuration options around prefixing the stack trace. You can:

-  customize the prefix and the exception message
-  keep the original exception message (don't scrub)
-  pass an allow_list of strings. Exception messages will be scrubbed unless the message or the
exception type regex match one of the allow_list strings.

```python
{!docs/compliant_logging/prefix-stack-trace.py!}
```

### With statements

Use this library with `with` statements:

```python
{!docs/compliant_logging/with-statement.py!}
```

### Directly with try / except statements

Using this library directly inside `try` / `except` statements:

```python
{!docs/compliant_logging/try-except.py!}
```

### Public exception types

Using the `Public*` exception types:

```python
{!docs/compliant_logging/public-exceptions.py!}
```

## Exception or Stack trace parsing

The `stack_trace_extractor` namespace contains simple tools to grab Python or C\#
stack traces and exceptions from log files. Sometimes the file that has the
stack trace you need may also contain sensitive data. Use this tool to parse and
print the stack trace, exception type and optionally exception message (careful
as  exception messages may also potentially hold private data.)

```python
from shrike.compliant_logging.stack_trace_extractor import StacktraceExtractor

extractor = StacktraceExtractor()
extractor.extract("log_file")
```
