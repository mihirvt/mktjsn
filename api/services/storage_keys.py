from api.constants import S3_KEY_PREFIX


def build_storage_key(key: str) -> str:
    """Build a storage key with optional configured prefix."""
    normalized_key = key.lstrip("/")
    if not S3_KEY_PREFIX:
        return normalized_key

    prefixed = f"{S3_KEY_PREFIX}/"
    if normalized_key.startswith(prefixed):
        return normalized_key

    return f"{prefixed}{normalized_key}"


def strip_storage_prefix(key: str) -> str:
    """Strip configured prefix from a storage key if present."""
    normalized_key = key.lstrip("/")
    if not S3_KEY_PREFIX:
        return normalized_key

    prefixed = f"{S3_KEY_PREFIX}/"
    if normalized_key.startswith(prefixed):
        return normalized_key[len(prefixed) :]

    return normalized_key
