"""Shared pytest setup.

Force a non-interactive matplotlib backend before anything in `optycal`
(transitively, via `emsutil`) has a chance to import pyplot with an
interactive backend, which would try to open a window / block on `.show()`
in a headless test run.
"""
import matplotlib

matplotlib.use("Agg")
