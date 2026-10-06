# Data sources

Tips for downloading a dataset from where it is published: what to pin, what to cache, what the source charges for, and which calls to avoid.
One section per kind of source, and so far every example starts from a Hugging Face dataset repo.
Read the matching section before writing `download.py` and the episode index.

## Hugging Face dataset repo

The listing driver is shared: `rrd_datasets_common/hf_repo.py`.
Each example adds its own episode index module on top of it.

- Pin — set `HF_REVISION` at the top of the episode index to a full commit sha, not a branch or a tag, so a re-upload cannot change what the converter reads.
  To pick up newly published episodes, bump the sha in its own commit.
- List and index — call `hf_file_index(repo_id, CACHE_PATH, HF_REVISION)` for every file path in the repo, cached at `examples/[dataset]/.cache/hf_files.json.gz` (gitignored) and keyed by the repo's commit sha.
  Build the episode index from that listing alone: name the files each episode needs, and settle flags such as `has_ir` from which sidecar files are next to it, so a worker never has to ask the Hub about an episode.
  Give the user a warning to expect a first listing of a large repo to take minutes.
  Later runs read the cache offline, since a pinned full sha needs no lookup.
  Delete the file to force a re-listing.
- Fetch — download with `hf_hub_download`, one file at a time at the pinned revision, which skips what is already on disk.
  Avoid `snapshot_download`: it drains the Hub rate limit, and the per-file calls cost nothing extra because the index already names every file.
- Gating and quota — for a gated repo, accept its terms while signed in to the Hub, then authenticate with that same account (`hf auth login` or `$HF_TOKEN`).
  For a public repo the credential is optional and only raises the quota, which anonymous callers share per IP.
  Pass the token to the Modal workers as an ephemeral per-run secret (`modal_jobs.image.hf_token_secret`), so nothing is stored on the runner.
  Put `HF_HUB_ENV` from `hf_repo.py` into the worker environment to turn off Xet and telemetry, so a download bills the roomier resolvers quota instead of the small api quota.
