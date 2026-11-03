"""Run ToolUniverse's MCP stdio server with a fix for compose tools that unload every other tool.

The matched profile starts it as

  uvx --from tooluniverse==<TOOLUNIVERSE_VERSION> python -P evaluator/tooluniverse_server.py

The MCP server registers tool names longer than 45 characters in shortened form, but a
ComposeTool checks its dependencies by full name. BiomarkerDiscoveryWorkflow, which the SLE task
recommends, therefore finds two HPA tools "missing" on every call and auto-loads their category
with load_tools(tool_type=[category]); that clears the registry by default, so the server keeps
156 of 2,797 tools and execute_tool is gone for the rest of the run.

The wrapper below treats dependencies registered under a shortened name as present, and loads
anything really missing without clearing the registry. Both of ComposeTool's other availability
checks go through _load_missing_dependencies when a name misses, so this covers all three.
Remove once the upstream fix is released (and TOOLUNIVERSE_VERSION moves past it).
"""
import functools
import importlib.metadata
import sys

TOOLUNIVERSE_VERSION = "1.5.8"


def patch_compose_tool():
    from tooluniverse.compose_tool import ComposeTool

    original = ComposeTool._load_missing_dependencies
    if getattr(original, "_opera_patched", False):
        return

    @functools.wraps(original)
    def load_missing_dependencies(self, missing_tools):
        universe = self.tooluniverse
        if universe is None:
            return original(self, missing_tools)
        registered = {name for name in missing_tools
                      if universe._resolve_tool_name(name) in universe.all_tool_dict}
        rest = set(missing_tools) - registered
        if not rest:
            return registered, set()
        universe.load_tools = functools.partial(type(universe).load_tools, universe,
                                                clear_existing=False)
        try:
            loaded, failed = original(self, rest)
        finally:
            del universe.load_tools
        return registered | loaded, failed

    load_missing_dependencies._opera_patched = True
    ComposeTool._load_missing_dependencies = load_missing_dependencies


def main():
    installed = importlib.metadata.version("tooluniverse")
    if installed != TOOLUNIVERSE_VERSION:
        sys.exit(f"tooluniverse {installed} is installed; this wrapper was checked against "
                 f"{TOOLUNIVERSE_VERSION}")
    patch_compose_tool()
    from tooluniverse.smcp_server import run_default_stdio_server
    run_default_stdio_server()


if __name__ == "__main__":
    main()
