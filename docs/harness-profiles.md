# Agent harness profiles

OPERA evaluates an agent *harness*: the model plus the CLI that runs it (scaffolding, built-in
tools, system prompt). How a run is launched therefore affects what is being measured. The run
launcher (`evaluator/run.py`, also installed as `bioagent-run`) supports two profiles.

## The two profiles

| | **pilot** | **matched** |
|---|---|---|
| Same command as | `scripts/run_eval.sh` | — |
| Working directory | Repo root | Clean per-run `workspace/`, no repo files |
| CLI configuration | The launching user's own (plugins, hooks, connectors, extra MCP servers, settings) | An empty config home per CLI (`CLAUDE_CONFIG_DIR`, `GEMINI_CLI_HOME`): what a fresh install gets |
| OPERA tools | `hypothesis-ledger` + `tooluniverse` from the repo's `.mcp.json` / `.gemini/settings.json`, plus anything in the user's configuration | Only `hypothesis-ledger` + `tooluniverse`, with absolute paths (`--strict-mcp-config` for Claude; the Gemini workspace is trusted) |
| Task delivery, Claude Code | **Replaces** the built-in system prompt (`--system-prompt`) | **Appended** to the built-in system prompt (`--append-system-prompt-file`) |
| Task delivery, Gemini CLI | In the user message (`Instruction: <task> ... Begin the task.`) | **Appended** to the built-in system prompt (a `GEMINI.md` in the workspace) |
| User message | Claude: `Begin the task.`; Gemini: the task | `Begin the task.` for both |
| Turn cap | Claude `--max-turns N`; Gemini none | Claude `--max-turns N`; Gemini `model.maxSessionTurns N` |
| Subagents | On (each CLI's default) | Off: Claude `--disallowedTools Agent Task`; Gemini `tools.exclude ["invoke_agent"]` |
| Claude background tasks | On | Off (`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`) |
| Stopping | When the CLI exits, or at `--timeout-minutes` | Also 15 s after the CLI's first result record |
| Task spec (rubric) | Copied into the run directory before the run | Copied in after the run |
| Auth | Whatever the user's CLI is configured with | API keys only |

Use **matched** to compare models or harnesses. The **pilot** profile results depend on the
launching user's CLI configuration, so two users (or the same user on two dates) can get different
setups from the same command.

## Why the matched profile looks this way

- **Fresh config homes, not Claude's `--bare` mode.** `--bare` also reduces Claude Code's built-in
  tools to Bash, Edit, and Read. An empty `CLAUDE_CONFIG_DIR` keeps the shipped tools (including web
  search, web fetch, and subagents) and bundled skills, and drops everything user-specific.
  `GEMINI_CLI_HOME` does the same for Gemini CLI.
- **Append, not replace.** Each harness keeps its built-in instructions (tool use, file editing), so
  each runs as shipped, with the task at the system level for both.
- **Rubric out of the run directory during the run.** The reviewer needs `task_spec.yaml` in the run
  directory, but the agent's workspace sits inside it, so the spec is copied in afterwards.
- **Gemini workspace trust.** With a fresh configuration, Gemini CLI disables MCP servers in
  untrusted folders even with `--skip-trust`; the launcher sets `GEMINI_CLI_TRUST_WORKSPACE=true`.
- **No secrets in run artifacts.** Tool API keys (`NCBI_API_KEY` etc.) are written into the
  configurations as `${VAR}` references, never values.
- **Subagents off.** Subagents do work the turn cap doesn't count, often on a smaller model. With
  them off, the cap covers all of the agent's work and the requested model does it. In Gemini CLI
  0.50, `experimental.enableAgents: false` only disables user-defined agents; the built-in ones
  stay reachable until the `invoke_agent` tool is excluded.
  `tools.exclude` is deprecated (Gemini CLI warns it will be removed in 1.0 in favor of its policy
  engine); recheck this when upgrading the CLI.
- **Claude Code background tasks off.** In `claude -p`, a background shell command that finishes
  after the turn cap starts a new invocation in the same session with a fresh turn allowance, and
  one that never exits keeps the process running. Gemini CLI's turn cap is a hard stop and its
  background processes don't wake the agent, so they are left on.
- **The launcher enforces the end of the run.** It stops a matched run that is still running 15 s
  after its first result record, stops any run at `--timeout-minutes` (default 120), and kills the
  whole process group, including MCP servers and leftover shell commands.

## Turn caps mean the same thing in both harnesses

Both CLIs count one **model response** as one turn, where a response may contain several tool
calls. Checked with a cap of 2: Claude Code made 2 responses, then stopped. It reports
`num_turns: 3`, because it counts the turn that hits the limit, so a cap of 30 reports 31. Gemini
CLI made 2 responses (4 parallel tool calls), then stopped with `FatalTurnLimitedError`. The same
cap therefore gives the same budget.

When comparing runs, measure budget in **main-agent model responses**. In pilot-profile runs,
Claude Code subagent responses are counted separately; the turn cap doesn't limit them, and a
Claude run can exceed the cap through background tasks (more than one session segment).

## The run manifest

Every run writes `run_manifest.json` next to `transcript.jsonl`:

- **setup:** profile, harness, CLI version, requested model, task delivery, turn cap, subagent and
  background-task settings, isolation, working directory, environment variables set, the command
  (task text replaced by a placeholder), SHA-256 of `agent_task.md`, code commit, whether tracked
  files were modified, and the stopping rules;
- **timing:** start and end times, exit code, and whether the launcher stopped the run
  (`stopped_by_launcher`: `after_first_result`, `timeout`, or null);
- **outcome:** session segments (more than one means the CLI started another invocation, so the
  total budget exceeds the cap), turns as reported by the CLI, main-agent and subagent model
  responses, tool calls, whether the cap was hit, and the models the provider reported with their
  input plus output token counts. (Gemini CLI's per-model `total_tokens` includes cached reads and
  over-reports them for subagent models.)

Check the models. A CLI can serve a different model than requested (for example, an alias
resolving to a newer model), and both CLIs use a smaller helper model for some internal work. The
launcher warns when most tokens did not go to the requested model.

## Limits

- **No container isolation.** The agent has shell access to the host and could find repository
  files, including task specs, by searching the filesystem. The clean workspace and configuration
  prevent accidental exposure, not a deliberate search. Use the `Dockerfile` for stronger isolation.
- **Harnesses still differ by design**: built-in tools, how MCP tools are loaded, and helper
  models. These are part of the harness being evaluated. With subagents off, neither CLI runs
  exactly as shipped.
- **CLI versions change.** Keep runs that will be compared close together in time; the manifest
  records each run's CLI version.

## Usage

```bash
set -a; source .env; set +a                                   # API keys
python -m evaluator.run claude-sonnet-4-6 --profile matched --dry-run   # show the plan, write nothing
python -m evaluator.run claude-sonnet-4-6 --profile matched --max-turns 30
python -m evaluator.run gemini-2.5-pro   --profile matched --max-turns 30
```

Options: `--task` (default `sle_biomarker_discovery`), `--runs-root` (default `~/bioeval/runs`, or
`$WORKSPACE_ROOT/runs`), `--run-id`, `--gemini-max-session-turns` (default: `--max-turns`),
`--timeout-minutes` (default 120; 0 for no limit), `--skip-log-replay`. Runs go to `<runs-root>/<model>_<profile>_<timestamp>/`.
