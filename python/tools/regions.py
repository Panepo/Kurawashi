from __future__ import annotations

import re


_STATE_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
_CODE_NAMES = {code.casefold(): code for code in _STATE_CODES.values()}
_CITY_STATES: dict[str, tuple[str, ...]] = {
    "anchorage": ("AK",),
    "atlanta": ("GA",),
    "austin": ("TX",),
    "baltimore": ("MD",),
    "bellevue": ("WA",),
    "bellingham": ("WA",),
    "boise": ("ID",),
    "boston": ("MA",),
    "charleston": ("SC", "WV"),
    "charlotte": ("NC",),
    "chicago": ("IL",),
    "cleveland": ("OH",),
    "colorado springs": ("CO",),
    "columbus": ("OH",),
    "dallas": ("TX",),
    "denver": ("CO",),
    "detroit": ("MI",),
    "el paso": ("TX",),
    "fort worth": ("TX",),
    "fresno": ("CA",),
    "honolulu": ("HI",),
    "houston": ("TX",),
    "indianapolis": ("IN",),
    "jacksonville": ("FL",),
    "kansas city": ("KS", "MO"),
    "las vegas": ("NV",),
    "los angeles": ("CA",),
    "madison": ("WI",),
    "memphis": ("TN",),
    "miami": ("FL",),
    "milwaukee": ("WI",),
    "minneapolis": ("MN",),
    "nashville": ("TN",),
    "new orleans": ("LA",),
    "new york city": ("NY",),
    "nyc": ("NY",),
    "oklahoma city": ("OK",),
    "omaha": ("NE",),
    "orlando": ("FL",),
    "philadelphia": ("PA",),
    "phoenix": ("AZ",),
    "pittsburgh": ("PA",),
    "portland": ("ME", "OR"),
    "raleigh": ("NC",),
    "reno": ("NV",),
    "richmond": ("VA",),
    "sacramento": ("CA",),
    "salt lake city": ("UT",),
    "san antonio": ("TX",),
    "san diego": ("CA",),
    "san francisco": ("CA",),
    "san jose": ("CA",),
    "savannah": ("GA",),
    "seattle": ("WA",),
    "spokane": ("WA",),
    "st louis": ("MO",),
    "st paul": ("MN",),
    "tacoma": ("WA",),
    "tampa": ("FL",),
    "tucson": ("AZ",),
}


def _place_key(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


def resolve_us_region(value: str) -> str:
    """Resolve a U.S. state name, USPS abbreviation, or curated city to a state code."""
    key = _place_key(value)
    if key in _STATE_CODES:
        return _STATE_CODES[key]
    if key in _CODE_NAMES:
        return _CODE_NAMES[key]
    states = _CITY_STATES.get(key)
    if states is None:
        raise ValueError(f"{value!r} is not a supported U.S. state or city.")
    if len(states) != 1:
        choices = ", ".join(states)
        raise ValueError(f"{value!r} is ambiguous; provide a state abbreviation ({choices}).")
    return states[0]


def us_region_is_mentioned(value: str, text: str) -> bool:
    """Check whether a canonical region is represented by a literal or supported alias."""
    expected = resolve_us_region(value)
    if re.search(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", text, re.IGNORECASE):
        return True
    for alias, code in _STATE_CODES.items():
        phrase = r"\s+".join(re.escape(part) for part in alias.split())
        pattern = r"(?<!\w)" + phrase + r"(?!\w)"
        if code == expected and re.search(pattern, text, re.IGNORECASE):
            return True
    for alias, states in _CITY_STATES.items():
        phrase = r"\s+".join(re.escape(part) for part in alias.split())
        pattern = r"(?<!\w)" + phrase + r"(?!\w)"
        if not re.search(pattern, text, re.IGNORECASE):
            continue
        if len(states) != 1:
            choices = ", ".join(states)
            raise ValueError(f"{alias!r} is ambiguous; provide a state abbreviation ({choices}).")
        if states[0] == expected:
            return True
    return False
