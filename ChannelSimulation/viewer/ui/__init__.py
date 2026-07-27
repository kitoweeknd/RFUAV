"""
UI layer: widgets and visualization components.
"""
from ChannelSimulation.viewer.ui.collapsible_section import CollapsibleSection
from ChannelSimulation.viewer.ui.signal_info_panel import SignalInfoPanel
from ChannelSimulation.viewer.ui.channel_config_panel import ChannelConfigPanel
from ChannelSimulation.viewer.ui.spectrum_viewer import SpectrumViewer
from ChannelSimulation.viewer.ui.snr_view import SNRView

__all__ = [
    "CollapsibleSection",
    "SignalInfoPanel",
    "ChannelConfigPanel",
    "SpectrumViewer",
    "SNRView",
]
