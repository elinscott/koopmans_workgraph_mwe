from typing import Literal, TypedDict


class CommandsConfig(TypedDict):
    kind: Literal["command_configs"]
    pw: str

def command_configs_factory(pw: str) -> CommandsConfig:
    return CommandsConfig(kind='command_configs', pw=pw)
