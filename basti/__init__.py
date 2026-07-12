"""BaSTI-IAC model support for the isochrones package (additive extension).

See models.py for the pseudo-EEP construction and cache-versioning notes.
"""
from .isochrone import get_ichrone_basti, Basti_Isochrone
from .models import BastiIsochroneGrid
