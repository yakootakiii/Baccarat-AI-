def payout(bet, outcome):
    """Profit in units for a 1-unit bet."""
    if bet == "B":
        return 0.95 if outcome == "B" else (0.0 if outcome == "T" else -1.0)
    if bet == "P":
        return 1.0 if outcome == "P" else (0.0 if outcome == "T" else -1.0)
    if bet == "T":
        return 8.0 if outcome == "T" else -1.0
    raise ValueError(bet)


def choose_bet(probs):
    """probs: {'B': p, 'P': p, 'T': p}. Picks the bet with the highest
    expected value. EV accounts for real payouts and ties pushing."""
    ev = {
        "B": 0.95 * probs["B"] - probs["P"],
        "P": probs["P"] - probs["B"],
        "T": 8 * probs["T"] - (1 - probs["T"]),
    }
    best = max(ev, key=ev.get)
    return best, ev
