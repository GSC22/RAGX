"""
Small shared helpers used across the API layer.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

# Matches exactly what generate_document_id() produces: 12 lowercase
# hex characters. Any document_id that doesn't match this shape did
# not come from our own generator, so it's rejected before it ever
# touches a filesystem path.
_DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def generate_document_id() -> str:
    """A short, URL-safe unique id assigned to each uploaded document.

    Truncated to 12 hex characters (48 bits) -- collisions are
    astronomically unlikely for a student project's scale of usage,
    and a shorter id is friendlier in URLs and API responses than a
    full 32-character UUID.
    """
    return uuid.uuid4().hex[:12]


def is_valid_document_id(document_id: str) -> bool:
    """Check that a document_id has the exact shape generate_document_id()
    produces, before it's used to build a filesystem path.

    This is defense in depth, not the only thing protecting against
    path traversal: every endpoint that takes a document_id also looks
    it up in the registry before touching disk, and an id that was
    never registered already 404s harmlessly. But relying solely on
    that ordering means a future code change (a new endpoint, a
    reordered check) could silently reopen the hole -- validating the
    id's *shape* up front removes that possibility entirely, rather
    than depending on every future contributor getting the order right.
    """
    return bool(_DOCUMENT_ID_PATTERN.match(document_id))


def sanitize_filename(filename: str) -> str:
    """Strip any path components and replace unsafe characters.

    This exists purely for security: an uploaded filename is
    attacker-controlled input. Without sanitizing it, a filename like
    "../../etc/passwd" or one containing null bytes/shell metacharacters
    could be used to write outside the intended uploads directory, or
    cause issues on some filesystems. `Path(filename).name` drops any
    directory components, and the regex whitelist keeps only
    characters that are safe everywhere (letters, digits, dot,
    underscore, hyphen).
    """
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "unnamed_file"

