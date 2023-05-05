import fsspec

S3PREFIX = "s3"


def get_filesystem(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem(S3PREFIX, anon=True) if path.startswith(S3PREFIX) else fsspec.filesystem("file")
