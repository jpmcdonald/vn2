"""Unit tests for szablowski.policy — analytical newsvendor ordering policy."""

import numpy as np
import pytest

from szablowski.policy import PolicyParams, compute_order


class TestPolicyParams:
    def test_critical_fractile(self):
        params = PolicyParams(cs=1.0, ch=0.2)
        assert params.critical_fractile == pytest.approx(1.0 / 1.2, rel=1e-6)

    def test_z_q_positive(self):
        params = PolicyParams(cs=1.0, ch=0.2)
        assert params.z_q > 0


class TestComputeOrder:
    def test_known_state(self):
        """With known inventory and forecasts, verify order is non-negative and reasonable."""
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (10.0, 10.0, 10.0)
        on_hand = 5
        in_transit = (5, 5, 0)
        order = compute_order(d_hat, on_hand, in_transit, params)
        assert order >= 0
        assert isinstance(order, int)

    def test_l3_projection_depletes_inventory(self):
        """L=3 projection: inventory depletes through h1, h2 before computing order for h3."""
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (10.0, 10.0, 10.0)
        on_hand = 30
        in_transit = (0, 0, 0)

        order = compute_order(d_hat, on_hand, in_transit, params)

        # on_hand=30, d1=10 -> leftover_t1=20
        # leftover_t1=20, d2=10 -> leftover_t2=10
        # projected_inv for t+3 = 10 + 0 = 10
        # target = 10 + z_q * phi * sqrt(10)
        z_q = params.z_q
        target = 10.0 + z_q * 1.0 * np.sqrt(10.0)
        expected = max(int(np.round(target - 10)), 0)
        assert order == expected

    def test_zero_demand_yields_zero_or_small_order(self):
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (0.0, 0.0, 0.0)
        on_hand = 10
        in_transit = (0, 0, 0)
        order = compute_order(d_hat, on_hand, in_transit, params)
        assert order == 0

    def test_zero_inventory_orders_to_target(self):
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (10.0, 10.0, 10.0)
        on_hand = 0
        in_transit = (0, 0, 0)
        order = compute_order(d_hat, on_hand, in_transit, params)
        # Everything depleted: projected_inv = 0
        z_q = params.z_q
        target = 10.0 + z_q * 1.0 * np.sqrt(10.0)
        expected = max(int(np.round(target)), 0)
        assert order == expected

    def test_phi_zero_yields_point_forecast_order(self):
        params = PolicyParams(cs=1.0, ch=0.2, phi=0.0)
        d_hat = (5.0, 5.0, 5.0)
        on_hand = 0
        in_transit = (0, 0, 0)
        order = compute_order(d_hat, on_hand, in_transit, params)
        # target = d3 + 0 = 5.0, projected_inv = 0
        assert order == 5

    def test_excess_inventory_yields_zero_order(self):
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (5.0, 5.0, 5.0)
        on_hand = 100
        in_transit = (0, 0, 0)
        order = compute_order(d_hat, on_hand, in_transit, params)
        assert order == 0

    def test_in_transit_reduces_order(self):
        params = PolicyParams(cs=1.0, ch=0.2, phi=1.0)
        d_hat = (10.0, 10.0, 10.0)
        on_hand = 0
        order_no_transit = compute_order(d_hat, on_hand, (0, 0, 0), params)
        # in_transit[0]=20 and [1]=20 exceed d1=10 and d2=10, leaving surplus
        # that flows into projected_inventory for t+3
        order_with_transit = compute_order(d_hat, on_hand, (20, 20, 0), params)
        assert order_with_transit < order_no_transit
