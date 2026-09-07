from football_pipeline.fixtures import fixtures_to_payloads


def test_fixtures_to_payloads_strips_scores_and_keeps_every_row():
    rows = [
        {
            "MatchNumber": 1,
            "RoundNumber": 1,
            "DateUtc": "2026-08-15 11:30:00Z",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Bournemouth",
            "HomeTeamScore": 4,
            "AwayTeamScore": 2,
        },
        {
            "MatchNumber": 40,
            "RoundNumber": 4,
            "DateUtc": "2026-09-12 14:00:00Z",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Man City",
            "HomeTeamScore": None,
            "AwayTeamScore": None,
        },
        {
            "MatchNumber": 41,
            "RoundNumber": 4,
            "DateUtc": "2026-09-13T16:30:00Z",
            "HomeTeam": "Spurs",
            "AwayTeam": "West Ham",
            "HomeTeamScore": "",
            "AwayTeamScore": "",
        },
    ]
    payloads = fixtures_to_payloads(rows)
    assert len(payloads) == 3
    assert payloads[0]["HomeTeam"] == "Liverpool"
    assert payloads[0]["FTHG"] == ""
    assert payloads[0]["FTR"] == ""
    assert payloads[1]["Date"] == "12/09/2026"
    assert payloads[1]["Time"] == "14:00"
    assert payloads[2]["HomeTeam"] == "Spurs"
    assert all(row["Div"] == "E0" and row["FTHG"] == "" for row in payloads)
