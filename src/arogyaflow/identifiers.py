from uuid import uuid4


def new_identifier(prefix: str | None = None) -> str:
    value = uuid4().hex
    return f"{prefix}_{value}" if prefix else value
