"""
DEPRECATED. The MCP surface has been split into two layered servers:

  - mcp_servers.openfda_device — data layer (find_devices, get_clearance)
  - mcp_servers.grounded_rag   — reasoning layer (ask, compare_performance, list_corpora)

This module is kept as a backward-compatible alias for the data-layer server so
`python -m ivd_mcp` still launches something useful. New code should target the
servers under mcp_servers/ directly.
"""

from __future__ import annotations

from mcp_servers.openfda_device.server import mcp, find_devices, get_clearance

__all__ = ["mcp", "find_devices", "get_clearance"]
