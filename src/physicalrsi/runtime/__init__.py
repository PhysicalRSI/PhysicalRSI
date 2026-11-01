"""Sensor-facing policy execution; imports never connect to hardware."""

from .sensors import store_rgbd

__all__ = ["store_rgbd"]
