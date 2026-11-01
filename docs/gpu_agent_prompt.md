# GPU Agent Startup Prompt

The README is self-contained. The following single instruction is sufficient when the agent has this repository, suitable GPU access, and repository/Release write access:

> Read README.md and execute the complete EEG stage, including training both models, validation, code delivery, and GitHub Release uploads of complete checkpoints. Verify downloaded artifacts and demonstrate inference and training resumption. Write all documentation and reports in English.

No separate reading of this file or `docs/eeg_stage.md` is required. The README contains the data sources, participant-split rules, preprocessing requirements, EEGNet and CBraMod training plans, calibration and evaluation protocol, prediction-table contract, checkpoint retention rules, and acceptance checklist.

Existing papers, proposal files, and raw data remain read-only. The assignment covers EEG implementation and experiments, code pushes, and Release delivery. It does not include SAC or robot-environment experiments.

Completion requires actual runs and verified remote artifacts. If GPU access, data semantics, repository permissions, or Release access blocks a step, report the evidence, preserve all artifacts, and finish unaffected work. Never report unexecuted training or unsuccessful uploads as complete.

The final handoff must include the repository URL, code commit, Release URL, metrics for both models, measured runtime, reproduction commands, checkpoint verification results, protocol deviations, and any unfinished work.
