# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pipeline helper class to create pipelines loading modules from a flexible manifest.
"""


from azure.ml.component import Component
from dataclasses import dataclass, field
import os
import logging
from typing import Optional, List

from shrike.pipeline.aml_connect import current_workspace


log = logging.getLogger(__name__)


@dataclass
class module_reference:
    key: Optional[
        str
    ] = None  # use as internal key to reference module (if None, use name)
    name: Optional[str] = None  # None if module exists only locally?
    source: Optional[str] = "registered"  # or "local"
    yaml: Optional[str] = None
    version: Optional[str] = None


@dataclass
class module_manifest:
    manifest: List[module_reference] = field(default_factory=list)


@dataclass
class module_loader_config:  # pylint: disable=invalid-name
    """Config for the AMLModuleLoader class"""

    use_local: Optional[str] = None
    use_local_except_for: Optional[bool] = None
    force_default_module_version: Optional[str] = None
    force_all_module_version: Optional[str] = None
    local_steps_folder: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "tests",
            "tests_pipeline",
            "sample",
            "steps",
        )
    )


class AMLModuleLoader:
    """Helper class to load modules from within an AMLPipelineHelper."""

    def __init__(self, config):
        """Creates module instances for AMLPipelineHelper.

        Args:
            config (DictConfig): configuration options
        """
        self.use_local_except_for = (
            config.module_loader.use_local_except_for
            if "use_local_except_for" in config.module_loader
            else None
        )
        if "use_local" not in config.module_loader:
            self.use_local = []
        elif config.module_loader.use_local is None:
            self.use_local = []
        elif config.module_loader.use_local == "*":
            self.use_local = "*"
        elif isinstance(config.module_loader.use_local, str):
            self.use_local = [
                x.strip() for x in config.module_loader.use_local.split(",")
            ]
            if not _check_use_local_syntax_valid(self.use_local):
                raise ValueError(
                    f'Invalid value for `use_local`. Please follow one of the four patterns: \n1) use_local="", all modules are remote\n2) use_local="*", all modules are local\n3) use_local="MODULE_KEY_1, MODULE_KEY_2", only MODULE_KEY_1, MODULE_KEY_2 are local, everything else is remote\n4) use_local="!MODULE_KEY_1, !MODULE_KEY_2", all except for MODULE_KEY_1, MODULE_KEY_2 are local'
                )
            self.use_local_except_for = self.use_local[0].startswith("!")

        self.force_default_module_version = (
            config.module_loader.force_default_module_version
            if "force_default_module_version" in config.module_loader
            else None
        )
        self.force_all_module_version = (
            config.module_loader.force_all_module_version
            if "force_all_module_version" in config.module_loader
            else None
        )
        self.local_steps_folder = config.module_loader.local_steps_folder
        self.module_cache = {}

        # internal manifest built from yaml config
        self.modules_manifest = {}
        self.load_config_manifest(config)

        log.info(
            f"AMLModuleLoader initialized (use_local={self.use_local}, force_default_module_version={self.force_default_module_version}, force_all_module_version={self.force_all_module_version}, local_steps_folder={self.local_steps_folder}, manifest={list(self.modules_manifest.keys())})"
        )

    def load_config_manifest(self, config):
        """Fills the internal module manifest based on config object"""
        for entry in config.modules.manifest:
            if entry.key:
                module_key = entry.key
            elif entry.name:
                module_key = entry.name
            else:
                raise Exception(
                    "In module manifest, you have to provide at least key or name."
                )

            self.modules_manifest[module_key] = entry

    def is_local(self, module_name):
        """Tests is module is in local list"""
        if self.use_local == "*":
            return True
        if self.use_local_except_for:
            return "!" + module_name not in self.use_local
        else:
            return module_name in self.use_local

    def module_in_cache(self, module_cache_key):
        """Tests if module in internal cache (dict)"""
        return module_cache_key in self.module_cache

    def get_from_cache(self, module_cache_key):
        """Gets module class from internal cache (dict)"""
        log.debug(f"Using cached module {module_cache_key}")
        return self.module_cache.get(module_cache_key, None)

    def put_in_cache(self, module_cache_key, module_class):
        """Puts module class in internal cache (dict)"""
        self.module_cache[module_cache_key] = module_class

    def verify_manifest(self, modules_manifest):
        """Tests a module manifest schema"""
        errors = []

        for (k, module_entry) in modules_manifest.items():
            # TODO: merge error checking code with processing code so we do all this in one pass
            if self.is_local(k):
                if "yaml_spec" not in module_entry:
                    errors.append(
                        f"{k}: You need to specify a yaml_spec for your module to use_local=['{k}']"
                    )
                elif not os.path.isfile(
                    module_entry["yaml_spec"]
                ) and not os.path.isfile(
                    os.path.join(self.local_steps_folder, module_entry["yaml_spec"])
                ):
                    errors.append(
                        "{}: Could not find yaml spec {} for use_local=['{}']".format(
                            k, module_entry["yaml_spec"], k
                        )
                    )
            else:
                if "remote_module_name" not in module_entry:
                    errors.append(
                        f"{k}: You need to specify a name for your module to use_local=False"
                    )
                if "namespace" not in module_entry:
                    errors.append(
                        f"{k}: You need to specify a namespace for your module to use_local=False"
                    )
                if ("version" not in module_entry) and (
                    self.force_default_module_version or self.force_all_module_version
                ):
                    errors.append(
                        f"{k}: You need to specify a version for your module to use_local=False, or use either force_default_module_version or force_all_module_version in config"
                    )

        return errors

    def load_local_module(self, module_spec_path):
        """Creates one module instance.

        Args:
            module_spec_path (str): path to local module yaml spec

        Returns:
            object: module class loaded
        """
        module_cache_key = module_spec_path
        if self.module_in_cache(module_cache_key):
            return self.get_from_cache(module_cache_key)

        log.info("Building module from local code at {}".format(module_spec_path))
        if not os.path.isfile(module_spec_path):
            module_spec_path = os.path.join(self.local_steps_folder, module_spec_path)
        loaded_module_class = Component.from_yaml(current_workspace(), module_spec_path)
        self.put_in_cache(module_cache_key, loaded_module_class)

        return loaded_module_class

    def load_prod_module(self, module_name, module_version, module_namespace=None):
        """Creates one module instance.

        Args:
            module_name (str) : module name
            module_version (str) : module version

        Returns:
            object: module class loaded
        """
        if self.force_all_module_version:
            module_version = self.force_all_module_version
        else:
            module_version = module_version or self.force_default_module_version

        module_cache_key = f"{module_name}:{module_version}"
        if self.module_in_cache(module_cache_key):
            return self.get_from_cache(module_cache_key)

        log.info(
            f"Loading remote module {module_cache_key} (name={module_name}, version={module_version}, namespace={module_namespace})"
        )
        loading_raised_exception = None

        try:
            # try without namespace first
            loaded_module_class = Component.load(
                current_workspace(),
                name=module_name,
                version=module_version,
            )
        except BaseException as e:
            # save the exception to raise it if namespace not provided
            if not module_namespace:
                raise e

        if module_namespace:
            log.info(
                f"    Trying to load module {module_name} with namespace {module_namespace}."
            )
            module_name = module_namespace + "://" + module_name
            loaded_module_class = Component.load(
                current_workspace(),
                name=module_name,
                version=module_version,
            )

        self.put_in_cache(module_cache_key, loaded_module_class)

        return loaded_module_class

    def get_module_manifest_entry(self, module_key, modules_manifest=None):
        """Gets a particular entry in the module manifest.

        Args:
            module_key (str): module key from the manifest
            modules_manifest (dict): manifest from required_modules() [DEPRECATED]

        Returns:
            dict: module manifest entry
        """
        if module_key in self.modules_manifest:
            module_entry = self.modules_manifest[module_key]
            module_namespace = None
        elif modules_manifest and module_key in modules_manifest:
            log.warning(
                f"We highly recommend substituting the `required_modules` method by the modules.manifest configuration."
            )
            module_entry = modules_manifest[module_key]
            # map to new format
            module_entry["yaml"] = module_entry["yaml_spec"]
            module_entry["name"] = module_entry["remote_module_name"]
            module_namespace = module_entry.get("namespace", None)
        else:
            raise Exception(
                f"Module key '{module_key}' could not be found in modules.manifest configuration or in required_modules() method."
            )

        return module_entry, module_namespace

    def load_module(self, module_key, modules_manifest=None):
        """Loads a particular module from the manifest.

        Args:
            module_key (str): module key from the manifest
            modules_manifest (dict): manifest from required_modules() [DEPRECATED]

        Returns:
            object: module class loaded
        """
        module_entry, module_namespace = self.get_module_manifest_entry(
            module_key, modules_manifest
        )

        if self.is_local(module_key):
            loaded_module = self.load_local_module(module_entry["yaml"])
        else:
            loaded_module = self.load_prod_module(
                module_entry["name"],
                module_entry["version"],
                module_namespace=module_namespace,
            )
        return loaded_module

    def load_modules_manifest(self, modules_manifest):
        """Creates module instances from modules_manifest.

        Args:
            modules_manifest (dict): manifest of modules to load

        Returns:
            dict: modules loaded, keys are taken from module_manifest.

        Raises:
            Exception: if loading module has an error or manifest is wrong.
        """
        log.info(f"Loading module manifest (use_local={self.use_local})")
        test_results = self.verify_manifest(modules_manifest)
        if test_results:
            raise Exception(
                "Loading modules from manifest raised errors:\n\nMANIFEST: {}\n\nERRORS: {}".format(
                    modules_manifest, "\n".join(test_results)
                )
            )

        loaded_modules = {}
        for module_key in modules_manifest:
            log.info(f"Loading module {module_key} from manifest")
            loaded_modules[module_key] = self.load_module(module_key, modules_manifest)

        return loaded_modules


def _check_use_local_syntax_valid(use_local_list) -> bool:
    use_local_except_for = True if use_local_list[0].startswith("!") else False
    if use_local_except_for:
        for module_key in use_local_list:
            if not module_key.startswith("!"):
                return False
    else:
        for module_key in use_local_list:
            if module_key.startswith("!"):
                return False
    return True
