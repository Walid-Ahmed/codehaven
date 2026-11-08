"""Booking fees."""
SERVICE_RATE = 0.05


def calc_fee(amount):
    return round(amount * SERVICE_RATE, 2)


def describe_fee(amount):
    return f"Service fee: {calc_fee(amount)}"
