# Examples

The simplest way to use this library is with the common pySpark entry script.
It will expect command line arguments `--zipFile` and `--binaryName`, the values
of which will be used to determine the archive and binary inside that archive
to invoke using the .NET runner. All other command line arguments are passed
directly to the compiled assembly.

```python
{!docs/spark/basic-use.py!}
```

:warning: The compiled C\# assembly is responsible for prefixing log statements
with `SystemLog:` and catching + prefixing exception stack traces. Long-term,
that functionality will be provided by
[This library could provide a C# version of the logging functionality](https://github.com/Azure/shrike/issues/62),
but for now you will have to home-brew your own.

It is possible to easily customize the command-line arguments for "zip file"
and "assembly name", e.g. like this.

```python
run_spark_net("--zip-file", "--assembly-name")
```

For more advanced configuration, e.g. customizing the Spark session, use the
`run_spark_net_from_known_assembly` method like below.

```python
{!docs/spark/advanced-use.py!}
```
