"""Hotel rooms and their availability."""
from bookings.fees import calc_fee

MAX_GUESTS = 4


class Room:
    def __init__(self, number, nightly_rate, under_maintenance=False):
        self.number = number
        self.nightly_rate = nightly_rate
        self.under_maintenance = under_maintenance
        self.booked_nights = set()

    def is_available(self, night):
        """Return True if the room can be booked that night; rooms under maintenance never can."""
        return night not in self.booked_nights

    def book(self, night, guests):
        if guests > MAX_GUESTS:
            raise ValueError("Too many guests")
        if not self.is_available(night):
            raise LookupError(f"Room {self.number} is taken on {night}")
        self.booked_nights.add(night)
        return self.nightly_rate + calc_fee(self.nightly_rate)
