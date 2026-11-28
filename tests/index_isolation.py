"""Keep the keyword index that tests build out of the real ~/CodeHaven/index.

A test module that indexes a repository with remembering on imports setUpModule and tearDownModule from here.
"""
import tempfile
from unittest.mock import patch

from offline_agent import index_cache

_state = {}


def setUpModule():
    """Point the default index folder at a temporary one for the whole module."""
    _state["temp"] = tempfile.TemporaryDirectory()
    _state["patch"] = patch.object(index_cache, "FOLDER", index_cache.Path(_state["temp"].name))
    _state["patch"].start()


def tearDownModule():
    """Restore the default index folder and delete what the module wrote."""
    _state.pop("patch").stop()
    _state.pop("temp").cleanup()
