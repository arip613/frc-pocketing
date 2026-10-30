from frcpocket.scrape.polite import parseContentSignals

chiefDelphiStyleRobots = """
# As a condition of accessing this website, you agree to abide by the following
# content signals:
User-agent: *
Content-Signal: search=yes,ai-train=no,use=reference
Allow: /

User-agent: GPTBot
Content-Signal: ai-train=yes
Disallow: /
"""


def test_parseContentSignals_reads_star_agent_only():
    signals = parseContentSignals(chiefDelphiStyleRobots)
    assert signals == {"search": "yes", "ai-train": "no", "use": "reference"}


def test_parseContentSignals_empty_when_absent():
    assert parseContentSignals("User-agent: *\nDisallow: /private\n") == {}
