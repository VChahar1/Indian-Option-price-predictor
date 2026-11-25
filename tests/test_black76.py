"""Offline identities for the forward-option pricing module."""

import math

from black76 import black76_price, implied_vol


def test_put_call_parity():
    for forward in (80, 100, 120):
        call = black76_price(forward, 100, 0.5, 0.25, 0.98, True)
        put = black76_price(forward, 100, 0.5, 0.25, 0.98, False)
        assert abs(call - put - 0.98 * (forward - 100)) < 1e-10


def test_implied_volatility_round_trip():
    for call in (True, False):
        for strike in (80, 100, 120):
            price = black76_price(100, strike, 0.5, 0.25, 0.98, call)
            recovered = implied_vol(price, 100, strike, 0.5, 0.98, call)
            assert abs(recovered - 0.25) < 1e-7


def test_expiry_intrinsic_and_invalid_iv():
    assert black76_price(110, 100, 0, 0.25, 0.98, True) == 9.8
    assert black76_price(90, 100, 0, 0.25, 0.98, True) == 0
    assert math.isnan(implied_vol(-1, 100, 100, 0.5, 0.98, True))
