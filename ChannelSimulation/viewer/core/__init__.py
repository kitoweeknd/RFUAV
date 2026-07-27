"""
Core utilities: styling, shortcuts, and general helpers.
"""
from ChannelSimulation.viewer.core.ui_style import THEME_LIGHT, THEME_DARK, get_all_stylesheet
from ChannelSimulation.viewer.core.shortcuts import SHORTCUTS
from ChannelSimulation.viewer.core.utils import (
    format_file_size,
    format_duration,
    parse_frequency_string,
    dict_merge,
)

__all__ = [
    "THEME_LIGHT",
    "THEME_DARK",
    "get_all_stylesheet",
    "SHORTCUTS",
    "format_file_size",
    "format_duration",
    "parse_frequency_string",
    "dict_merge",
]
