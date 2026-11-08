import unittest
from bookings.rooms import Room


class RoomTests(unittest.TestCase):
    def test_guest_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            Room(1, 100).book("2027-01-01", 5)
