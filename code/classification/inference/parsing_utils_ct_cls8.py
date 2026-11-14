"""OmegaConf 2.0-compatible resolver registration for CT_CLS8."""
from uuid import uuid4

from omegaconf import OmegaConf


def make_omegaconf_resolvers():
    OmegaConf.register_resolver(
        "path_formatter",
        lambda s: s.replace("[", "")
        .replace("]", "")
        .replace("}", "")
        .replace("{", "")
        .replace(")", "")
        .replace("(", "")
        .replace(",", "_")
        .replace("=", "_")
        .replace("/", ".")
        .replace("+", "")
        .replace("@", "."),
    )
    OmegaConf.register_resolver("model_name_extractor", lambda s: s.split(".")[-1])
    OmegaConf.register_resolver("make_group_name", lambda: str(uuid4()))
