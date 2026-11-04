# Running the MIRAGE benchmarks on the GPU server

Tested on CPU end to end (tiny model, small index). These are the steps for the A100.

**Budget:**
- **GPU:** a dedicated 40 GB A100. vLLM takes 85% (about 34 GB): Llama-3.1-8B in bf16 is 16 GB, and the rest is a large KV cache for about 128 parallel requests. The MedCPT retrieval models use about 1 GB of the remainder.
- **Disk:** about 40 GB in total (models about 17 GB, Python environments about 12 GB, corpora and indexes about 8 GB).

**Nothing secret is saved on the server.** The GitHub and Hugging Face tokens are only typed into hidden prompts or held in a variable for a single command. They never land in:
- shell history
- `.git/config`
- `~/.git-credentials`
- `~/.cache/huggingface/token`

Don't run `huggingface-cli login`: it writes the token to disk.

## 0. One-time setup

### 0a. On your laptop (browser)

1. **Accept the Llama licence.** Open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, log in and accept the Llama 3.1 Community License. Wait until the page says you've been granted access, which usually takes minutes.
2. **Create a Hugging Face token.** Go to https://huggingface.co/settings/tokens → *Create new token* → **Fine-grained** → tick only *Read access to contents of all public gated repos you can access*. Delete it after the download in step 0d.
3. **Create a GitHub token.** Go to https://github.com/settings/personal-access-tokens → *Generate new token* (fine-grained):
   - Repository access: only `Better-Clinical-Agent`
   - Permissions: *Contents → Read-only*
   - Expiration: 7 days

   (Alternatively, skip the token and use SSH agent forwarding, see the note under 0b.)
4. **Copy the patch to the server:**

```bash
scp phase1_rag.patch <user>@<server>:/path/to/big/disk/
```

### 0b. Clone and apply the patch (server)

```bash
export WORK=/path/to/big/disk/clinical-agent      # ~40 GB free
mkdir -p $WORK && cd $WORK
mv /path/to/big/disk/phase1_rag.patch $WORK/

# Username: your GitHub username.  Password: paste the GitHub token (it is not echoed).
# "-c credential.helper=" makes sure no credential helper stores it.
git -c credential.helper= clone https://github.com/<OWNER>/Better-Clinical-Agent.git
cd Better-Clinical-Agent

git apply --check ../phase1_rag.patch        # dry run, prints nothing if OK
git apply --index ../phase1_rag.patch
git -c user.name="<your name>" -c user.email="<your email>" commit -m "Phase 1: medical RAG + MIRAGE harness"
```

- The patch touches no files under `docs/presentations/`, `Papers/` or `paper_catalog/`.
- If `--check` complains, use `git apply --3way --index ../phase1_rag.patch` instead.
- *SSH alternative (no token at all):* connect with `ssh -A <user>@<server>`. Your laptop's SSH key is then used through the connection but never copied to the server. Clone with `git clone git@github.com:<OWNER>/Better-Clinical-Agent.git`.

*Starting from a downloaded zip instead of a clone* (e.g. while waiting for repo access):
commit the untouched zip contents first as a "baseline" commit, then apply the patch on top.
Later you can replay the patch commit onto the real history:

```bash
git remote add origin <repo-url> && git fetch origin
git rebase --onto origin/main <baseline-commit> <your-branch>
```

### 0c. Python environments with uv (server)

We use [uv](https://docs.astral.sh/uv/) rather than pip or conda:
- It installs packages 10-100x faster.
- It can download its own Python, so it doesn't depend on conda.
- vLLM recommends it: `--torch-backend=auto` picks the torch build that matches your CUDA driver.

If you have conda installed, run `conda deactivate` first so the two don't mix.

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # user-local install, no root
cd $WORK/Better-Clinical-Agent
nvidia-smi                                   # should show the A100 with ~40 GB free

# project environment
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt --torch-backend=cu126

# vLLM in its OWN environment (it pins its own torch version)
uv venv $WORK/vllm-env --python 3.12
uv pip install --python $WORK/vllm-env -r requirements-vllm.txt --torch-backend=cu128
```

**Match torch to the GPU driver.** Look at the "CUDA Version" in the top-right corner of `nvidia-smi`.

On our server it showed **12.2** (driver 535), which can only run CUDA 12.x builds. The default PyPI wheels are CUDA 13 and fail in two different ways:
- **The project's torch silently reports no GPU.** `build_index.py` now refuses to run in that case instead of encoding on CPU for hours.
- **vLLM fails with `libcudart.so.13`.** That is why `requirements-vllm.txt` pins `vllm==0.11.0` (the CUDA 12 line) and `transformers<5`.

Check both environments. Each must end with `True`:

```bash
.venv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
$WORK/vllm-env/bin/python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On a CUDA 13 driver, use `--torch-backend=cu130` and the latest `vllm` instead.

Then create two small settings files. Neither contains a secret:

```bash
# sourced in every new shell / tmux window: keeps all model downloads on the big disk
echo "export WORK=$WORK" > $WORK/env.sh
echo "export HF_HOME=$WORK/hf-cache" >> $WORK/env.sh

# read by the project code
echo "LLM_BASE_URL=http://localhost:8000/v1" > .env
echo "LLM_MODEL=llama31-8b" >> .env
echo "DATA_DIR=$WORK/data" >> .env
```

### 0d. Download Llama-3.1-8B once (server)

The token is read silently into a variable, used for this one download, then erased.

```bash
source $WORK/env.sh
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', ignore_patterns=['original/*'])"
unset HF_TOKEN
```

The weights (about 16 GB) are now in `$WORK/hf-cache`, and no token is needed again. You can delete the Hugging Face token on the website. Everything else (MedCPT, the corpora, MIRAGE) is public.

## 1. Build the knowledge indexes (once, ~20-30 min)

```bash
cd $WORK/Better-Clinical-Agent && source $WORK/env.sh && source .venv/bin/activate
python scripts/build_index.py download --corpora textbooks statpearls   # ~2.2 GB download
python scripts/build_index.py build --corpora textbooks statpearls      # GPU, ~10-20 min
```

Expected output: `Corpus: ~427,000 snippets`, then a BM25 time and a MedCPT time.

## 2. Start the LLM server (in tmux, so it survives disconnects)

```bash
tmux new -s vllm
cd $WORK/Better-Clinical-Agent && source $WORK/env.sh
export HF_HUB_OFFLINE=1                      # use the cached weights, never contact Hugging Face
PATH=$WORK/vllm-env/bin:$PATH bash scripts/serve_vllm.sh
# wait for "Application startup complete", then detach with Ctrl-b d
```

Check it from your normal shell:

```bash
curl -s localhost:8000/v1/models
```

If someone else needs the GPU again, share it with `GPU_UTIL=0.45 QUANT=fp8` in front of the serve command.

## 3. Sanity run (~5 minutes): send me this table first

```bash
cd $WORK/Better-Clinical-Agent && source $WORK/env.sh && source .venv/bin/activate
python scripts/run_mirage.py --dataset medqa --limit 50 --methods cot medrag ddx_rag --run-name sanity
```

## 4. Main runs (in a second tmux window: `tmux new -s runs`)

```bash
# tuning only, on MedQA *train* questions (never tune on test)
python scripts/run_mirage.py --dataset medqa_dev --methods cot ddx_rag

# headline comparison
python scripts/run_mirage.py --dataset medqa --methods cot cot_sc medrag rationale_rag imedrag ddx_rag
python scripts/run_mirage.py --dataset mmlu  --methods cot cot_sc medrag rationale_rag imedrag ddx_rag
```

## 5. Ablations and robustness (MedQA)

```bash
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set intents=SUPPORT,DISCRIMINATE --tag no_refute
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set intents=SUPPORT              --tag support_only
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set verify=false                 --tag no_verify
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set show_votes=false             --tag no_votes
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set gate_threshold=1.01          --tag no_gate
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set mode=dense                   --tag dense
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --set expand=false                 --tag no_expand
python scripts/run_mirage.py --dataset medqa --methods cot ddx_rag --shuffle-options 1 --run-name llama31-8b_shuffled
python scripts/run_mirage.py --dataset medqa --methods ddx_rag --noise 3 --tag noise3
```

## 5b. Optional: a stronger, medical second model

Repeat the headline comparison with MedGemma-27B (FP8, about 27 GB) to check whether the gain holds on a medical model.
1. Accept its licence at https://huggingface.co/google/medgemma-27b-text-it.
2. Download it the same way as in 0d, replacing the model name.
3. Stop the Llama server, then:

```bash
HF_HUB_OFFLINE=1 MODEL=google/medgemma-27b-text-it NAME=medgemma-27b QUANT=fp8 GPU_UTIL=0.88 MAX_SEQS=64 \
  PATH=$WORK/vllm-env/bin:$PATH bash scripts/serve_vllm.sh
LLM_MODEL=medgemma-27b python scripts/run_mirage.py --dataset medqa --methods cot medrag ddx_rag --workers 32
```

Results go to `results/medgemma-27b/`, separate from the Llama runs.

## 6. What to send back

```bash
python scripts/run_mirage.py --dataset medqa --report-only > report_medqa.md
python scripts/run_mirage.py --dataset mmlu  --report-only > report_mmlu.md
python scripts/analyze_results.py --dataset medqa > analysis_medqa.md
python scripts/analyze_results.py --dataset mmlu  > analysis_mmlu.md
tar czf results.tgz results/ report_*.md analysis_*.md
```

`analyze_results.py` looks inside the traces:
- whether vote agreement predicts correctness
- light vs. full path accuracy
- good vs. bad overrules of the closed-book majority
- which query intents supplied the evidence
- per-subject accuracy
- the exact questions each method fixed or broke

The `.jsonl` files contain the full traces (queries, evidence ids, outputs), which I need for error analysis.

## Notes

- **Resumable.** If a run dies, rerun the same command and it continues where it stopped.
- **Throughput.**
  - `--workers 48` (the default) keeps 48 questions in flight. With DDx-RAG's parallel votes that is more than 100 sequences, which matches `--max-num-seqs 128`.
  - If `nvidia-smi` shows low utilisation, try `--workers 64`.
  - Prefix caching is on: the 5 votes and the later steps of a question share the same long prompt prefix, so vLLM computes it once.
- **Rough cost** with an 8B model:
  - `cot`: a few minutes per dataset
  - `ddx_rag`: about 4× `cot`
  - `imedrag`: the most expensive, at about 13 calls per question
- **Errors.** Per-question errors are recorded in the `error` field, not raised. If the table shows many errors, look at those first.
- **Cleanup.** Nothing to revoke on the server. Delete or let expire the two tokens on GitHub and Hugging Face.
