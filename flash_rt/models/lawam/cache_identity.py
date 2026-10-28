"""Content identities for the loaded LaWAM checkpoint and processor."""
from pathlib import Path
import hashlib
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processor_sha256(processor):
    # Serialize the actual loaded processor, including added special tokens,
    # tokenizer vocabulary/merges, chat templates and image/video configuration.
    # A repository name or directory path is not an identity.
    with tempfile.TemporaryDirectory(prefix="lawam-processor-") as directory:
        processor.save_pretrained(directory)
        root = Path(directory)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if not files:
            raise ValueError("LaWAM processor did not serialize any content")
        digest = hashlib.sha256()
        for path in files:
            name = path.relative_to(root).as_posix().encode()
            digest.update(len(name).to_bytes(8, "little"))
            digest.update(name)
            digest.update(bytes.fromhex(file_sha256(path)))
        return digest.hexdigest()
