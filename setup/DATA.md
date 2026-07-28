# Downloading the data

Action Atlas is split into a light visualization bundle (what the website needs) and heavier
reproduction data (SAE weights, concepts, trajectories, raw activations). The downloader at
`setup/download_data.py` fetches either, in whole or in part, and places everything at the relative
paths the app and the experiment scripts already use. No code or path edits are needed afterward.

## Run the website locally

The frontend reads only the visualization bundle plus videos that stream from a public bucket. Fetch
the bundle and start the site:

```bash
python setup/download_data.py --viz
```

This downloads about 0.4 GB into `action_atlas/data/` (`processed/`, `feature_embeddings/`,
`descriptions/`). Without this step a fresh clone has the lighter index JSONs but not the clustering
and embedding data the frontend renders, so this is the one required download for the site.

## Download a subset

Pick any combination of models, artifact types, SAE pooling arms, and LIBERO suites.

```bash
# Per-token SAEs for one model (the primary release arm)
python setup/download_data.py --models pi05 --types saes --arms per_token

# SAEs plus concepts and trajectories for several models
python setup/download_data.py --models oft,xvla --types saes,concepts,trajectories

# Raw activations for one model, limited to one suite (heavy)
python setup/download_data.py --models oft --types activations --suites libero_goal

# Everything for every model
python setup/download_data.py --models all --types saes,concepts,trajectories,activations --arms all
```

`python setup/download_data.py --list` prints the full set of repos and where each lands.

## Where things land

| Selection | Local path |
|---|---|
| `--viz` | `action_atlas/data/` |
| `--types saes,concepts,trajectories` | `outputs/release/<model>/` |
| `--types activations` | `activations/<model>/` |

## Matching SAEs to activations

Every SAE carries an `activation_layer` field in its safetensors metadata and in the bundle
`manifest.jsonl`, naming the exact activation it maps to. The convention differs per model:

| Model | activation identifier | meaning |
|---|---|---|
| oft | `layer_{N}` | key inside each episode `.pt` |
| pi05 | `{pathway}/layer{NN}` | npz directory; array `activations` |
| xvla | `{env}_all_{pool}/activations/layer_{N}` | per-environment, per-pooling cache |
| smolvla | `{pathway}_L{NN}` | array key inside each episode `.npz` |
| groot | `{prefix}_L{NN}` | layer directory (`dit`, `eagle_lm`, `vl_sa`) |

Per-token is the primary, verified arm. Mean-pool and temporal are included for transparency; the
temporal arm is the InfoNCE ablation that degrades rollout fidelity by design, so it is labeled
`pooling=temporal` and is not intended as a usable feature extractor.

## Rollout-reconstruction protocols

Each SAE arm must be applied at rollout time with the same protocol it was trained under. The
`mean_pool` checkpoints were trained on token-axis mean-pooled vectors (`h.mean(dim=1)`), so
running them through a per-token applier is a silent train/eval mismatch that corrupts the
reconstruction. Use `experiments/reconstruction_hooks.py::make_reconstruction_hook`, which
implements all three protocols and refuses the mismatched pairing unless explicitly overridden
(`allow_mismatched_protocol=True`).

| Arm | Architecture | Protocol | What the hook does |
|---|---|---|---|
| per_token | all models | `pertoken_replace` | encode/decode every token position independently; full replace |
| mean_pool | pi0.5 (and other single-stream models) | `meanpool_replace` | pool tokens -> reconstruct the pooled vector -> broadcast-replace at every position |
| mean_pool | xvla | `meanpool_delta` | pool tokens -> reconstruct -> add the delta (`recon_mean - mean`) at every position |

X-VLA uses delta-add rather than broadcast-replace because its sequences are heterogeneous token
streams: broadcast-replacing every position would flatten per-token structure downstream layers
depend on, while adding the pooled reconstruction delta reconstructs the mean component and
preserves per-token variation.

## Reproducing experiments

After downloading SAEs and activations, the experiment scripts in `experiments/` consume them from
the paths above. See the project README for the baseline, SAE training, concept identification, and
cross-task injection commands.

The backend and scripts resolve local data under a configurable root. Set `ACTION_ATLAS_DATA_ROOT`
to point at wherever you placed the downloaded data; it defaults to `data` relative to the working
directory. Nothing is read from an absolute machine path.
