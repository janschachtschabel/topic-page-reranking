"""Turn whatever the user pastes into a collection id.

The pasted URL is only *parsed*, never fetched -- the app talks to the one
repository it is configured for, so a URL cannot point it anywhere else.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
#: Query parameters that name the node in edu-sharing and WLO links.
_ID_PARAMS = ("id", "collectionId", "collection", "nodeId", "node")


def collection_id_from(text: str) -> str:
    """The node id in a topic-page URL, or a bare id.

    A known query parameter wins; otherwise the last UUID in the URL, since
    paths name parents before children.

    Raises:
        ValueError: when the text holds no UUID.
    """
    text = (text or "").strip()
    query = parse_qs(urlparse(text).query)
    for name in _ID_PARAMS:
        for value in query.get(name, []):
            if found := UUID.search(value):
                return found.group(0).lower()
    found = UUID.findall(text)
    if not found:
        raise ValueError("Keine Knoten-ID (UUID) in der Eingabe gefunden.")
    return found[-1].lower()
