"""Display-only club badges. Not used for training or features."""

from __future__ import annotations

# football-data.org crest IDs. Missing/unknown clubs fall back to initials.
_CREST_ID = {
    "Arsenal": 57,
    "Aston Villa": 58,
    "Bournemouth": 1044,
    "Brentford": 402,
    "Brighton & Hove Albion": 397,
    "Burnley": 328,
    "Cardiff": 715,
    "Chelsea": 61,
    "Coventry City": 1079,
    "Crystal Palace": 354,
    "Everton": 62,
    "Fulham": 63,
    "Huddersfield": 394,
    "Hull City": 322,
    "Ipswich Town": 349,
    "Leeds United": 341,
    "Leicester": 338,
    "Liverpool": 64,
    "Luton": 389,
    "Manchester City": 65,
    "Manchester United": 66,
    "Newcastle United": 67,
    "Norwich": 68,
    "Nottingham Forest": 351,
    "Sheffield United": 356,
    "Southampton": 340,
    "Sunderland": 71,
    "Tottenham Hotspur": 73,
    "Watford": 346,
    "West Brom": 74,
    "West Ham United": 563,
    "Wolverhampton Wanderers": 76,
}

_COLORS = {
    "Arsenal": "#ef0107",
    "Aston Villa": "#95bfe5",
    "Bournemouth": "#da291c",
    "Brentford": "#e30613",
    "Brighton & Hove Albion": "#005daa",
    "Burnley": "#6c1d45",
    "Cardiff": "#0070b5",
    "Chelsea": "#034694",
    "Coventry City": "#1c9ad6",
    "Crystal Palace": "#1b458f",
    "Everton": "#003399",
    "Fulham": "#9ea7b1",
    "Huddersfield": "#0e63ad",
    "Hull City": "#f5a12d",
    "Ipswich Town": "#0044aa",
    "Leeds United": "#ffcd00",
    "Leicester": "#003090",
    "Liverpool": "#c8102e",
    "Luton": "#f78f1e",
    "Manchester City": "#6cabdd",
    "Manchester United": "#da291c",
    "Newcastle United": "#241f20",
    "Norwich": "#00a650",
    "Nottingham Forest": "#e53233",
    "Sheffield United": "#ee2737",
    "Southampton": "#d71920",
    "Sunderland": "#eb172b",
    "Tottenham Hotspur": "#132257",
    "Watford": "#fbee23",
    "West Brom": "#122f67",
    "West Ham United": "#7a263a",
    "Wolverhampton Wanderers": "#fdb913",
}

_INK_ON_LIGHT = {
    "Aston Villa",
    "Brighton & Hove Albion",
    "Fulham",
    "Hull City",
    "Leeds United",
    "Luton",
    "Manchester City",
    "Norwich",
    "Watford",
    "Wolverhampton Wanderers",
}


def team_initials(name: str) -> str:
    parts = [part for part in str(name).replace("&", " ").split() if part and part.lower() not in {"and", "the"}]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:3].upper()
    return "".join(part[0] for part in parts[:3]).upper()


def team_badge(name: str) -> dict:
    """Public crest URL plus a local initials fallback. Never invents match stats."""
    crest_id = _CREST_ID.get(name)
    color = _COLORS.get(name, "#3d4338")
    return {
        "crest_url": f"https://crests.football-data.org/{crest_id}.png" if crest_id else None,
        "initials": team_initials(name),
        "color": color,
        "ink": "#1a1408" if name in _INK_ON_LIGHT else "#f4f1e8",
    }
