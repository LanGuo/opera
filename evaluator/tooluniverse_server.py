"""Run ToolUniverse's MCP stdio server at the pinned version.

The matched profile starts it as

  uvx --python 3.12 --from tooluniverse==<TOOLUNIVERSE_VERSION> --with cellxgene-census \
      python -P evaluator/tooluniverse_server.py

1.5.8, used for the Tier 2 runs, needed a patch here: a compose tool such as
BiomarkerDiscoveryWorkflow found its long-named dependencies "missing" and reloaded its category
with the registry cleared, leaving 156 of 2,797 tools and no execute_tool. Our fix
(mims-harvard/ToolUniverse#845) is in 1.5.9 and later, so the patch is gone (D-37).

cellxgene-census goes into the same environment because ToolUniverse's CELLxGENE_* tools (expression
data, cell and gene metadata) need it, and ToolUniverse only offers them when
CELLXGENE_CENSUS_PACKAGE_INSTALLED=1 is set in its environment.
"""
import importlib.metadata
import sys

TOOLUNIVERSE_VERSION = "1.5.10"
EXTRA_PACKAGES = ("cellxgene-census",)
# ToolUniverse hides its CELLxGENE_* tools unless this "key" is set; it means the package is installed.
EXTRA_ENV = {"CELLXGENE_CENSUS_PACKAGE_INSTALLED": "1"}
PYTHON = "3.12"  # tiledbsoma, under cellxgene-census, doesn't build on 3.14


def main():
    installed = importlib.metadata.version("tooluniverse")
    if installed != TOOLUNIVERSE_VERSION:
        sys.exit(f"tooluniverse {installed} is installed; OPERA pins {TOOLUNIVERSE_VERSION}")
    from tooluniverse.smcp_server import run_default_stdio_server
    run_default_stdio_server()


if __name__ == "__main__":
    main()
