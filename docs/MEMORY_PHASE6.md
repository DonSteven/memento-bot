# Offline Memory Evaluation

`nanobot memory-eval --output json` runs deterministic fixture replay against the
structured memory path. `--save-dir` writes generated reports to a local directory.
Report files and external benchmark datasets are not repository inputs.

`nanobot memory-v2-eval` measures semantic extraction and preservation with its
configured embedding and judge backends. `nanobot memory-v2-query-eval` evaluates
snapshot or replay queries; supply an explicit `--cases-path` because external
query datasets are not bundled. These live evaluation commands can call the
configured model services. Their results are distinct from scripted unit tests.

For offline contract checks, run the tests in `nanobot/tests/phase_6`,
`nanobot/tests/phase_7` and `nanobot/tests/phase_8`. Tests use temporary workspaces,
fictional query cases and scripted providers. They validate evaluation behavior,
not real-model performance.
