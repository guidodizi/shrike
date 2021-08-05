# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

log = logging.getLogger(__name__)


class TelemetryLogger:
    """Utils class for opencensus azure monitor"""

    def __init__(
        self, enable_telemetry=True, instrumentation_key=None, level=logging.INFO
    ):
        self.logger = logging.getLogger("telemetry_logger")
        self.logger.setLevel(level)
        self.enable_telemetry = enable_telemetry
        # Why is it okay to include this key directly in the source code?
        # For any client-side tool, there is a fundamental problem with protecting instrumentation
        # keys. You want the published tool to be able to collect telemetry, but the only way
        # it can do this is if it has some kind of instrumentation key.
        #
        # For an authoritative example, the dotnet CLI contains their telemetry key in a
        # public GitHub repository:
        # https://github.com/dotnet/cli/blob/master/src/dotnet/Telemetry/Telemetry.cs
        #
        # The underlying Azure resource is called `aml1p-ml-tooling`.
        self.instrumentation_key = (
            "aaefce9e-d109-4fac-bb9f-8277c68e91ac"
            if instrumentation_key is None
            else instrumentation_key
        )
        handler = AzureLogHandler(
            connection_string=f"InstrumentationKey={self.instrumentation_key}"
        )
        handler.add_telemetry_processor(self.scrubber_function)
        self.logger.addHandler(handler)

    def log_trace(self, message, properties={}, level=logging.INFO):
        if self.enable_telemetry:
            try:
                if level == logging.INFO:
                    self.logger.info(message, extra=properties)
                elif level == logging.WARNING:
                    self.logger.warning(message, extra=properties)
                elif level == logging.ERROR:
                    self.logger.error(message, extra=properties)
                elif level == logging.CRITICAL:
                    self.logger.critical(message, extra=properties)
                else:
                    log.error("The logging level is not expected!")
            except Exception as ex:
                log.warning("Send telemetry exception: %s", str(ex))
        else:
            log.info(
                "Sending trace log messages to application insight has been disabled."
            )

    # Callback function to scrub some columns
    def scrubber_function(self, envelope):
        envelope.tags["ai.cloud.roleInstance"] = "cloud_RoleInstance_Scrubbed"
        envelope.tags["ai.location.ip"] = "IP_Scrubbed"
