"""Self-contained April 2026 experiment package for the current BrowseCompV2 matrix.

The package intentionally contains only the four experiment families that are active in the
current evaluation cycle:
- prose merge2
- structured merge2
- docsummaryaux merge2
- official RLM with prompt-doc companions from docsummaryaux

The code is organized so a collaborator can run those experiments locally without digging
through older shell wrappers, sharding scripts, or cluster-specific launch files.
"""

__all__ = ["common", "data", "methods"]
__version__ = "0.1.0"
