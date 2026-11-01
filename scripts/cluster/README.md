# Cluster job scripts (GM Warren LSF)

Exactly the scripts that ran the reported jobs. Paths are absolute to the project filer
(`/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg`); adapt `P` for another machine.

| script | queue | purpose |
|---|---|---|
| `setup_cpu.sh` | cpu_ubuntu_wrn | clone dataset + official repos, build env, download CBraMod weights |
| `smoke.sh`, `smoke2.sh` | gpu_large_shared (1 GPU, gc175) | smoke tests, benchmark, resume equivalence, supplementary smoke |
| `job_grid.sh` + `job_env.sh` | gpu_large_shared (1 GPU, gc175) | train all grid candidates of one or more configs |
| `job_post.sh` | gpu_large_shared (1 GPU, gc175) | selection, evaluation, extra seeds, supplementary evaluation, export |
| `job_verify.sh` | gpu_large_shared (1 GPU, gc175) | verify downloaded release assets, reproduce predictions, resume from last.ckpt |

Every GPU job requests exactly one GPU and ends when its work ends (no reservation is held).
