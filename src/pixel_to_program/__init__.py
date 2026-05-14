"""Synthetic visual programs with an optional trainable encoder/decoder."""

from .scene import Scene, Shape, parse_program, render_rgb, render_svg, serialize

__all__ = ["Scene", "Shape", "parse_program", "render_rgb", "render_svg", "serialize"]
