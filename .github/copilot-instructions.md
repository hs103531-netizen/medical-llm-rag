# AI Assistant Instructions for `medical_llm` Repository

This repo is a small pipeline for preparing Korean medical QA data from AI Hub and
finetuning a LLM using Colab/T4. The goal of any coding agent working here is to
become immediately productive by understanding how the pieces fit together and
by following the established conventions.

---

## 🎯 Big Picture

1. **Source**: AI Hub provides several Korean medical datasets (consultation QA,
summary, etc.). The data are downloaded manually and unpacked under
`data/raw/`.
2. **Pipeline**: three sequential Python scripts (prefixes 01–03) convert raw JSON
into the JSONL format required by LLM fine‑tuning.
3. **Training**: a Colab notebook (`04_finetune_T4_colab.ipynb`) is used for
QLoRA training with T4‑optimised settings; the notebook expects the
output of the previous script on Google Drive.

All scripts are stand‑alone CLI tools; they print progress and are designed to be
run from the repository root.

---

## 🗂 Data Flow and Directories

- `data/raw/` – downloaded AI Hub archives or sample files.
- `data/processed/` – cleaned JSON produced by `02_preprocessor.py`.
- `data/jsonl/` – final training splits (`train.jsonl`, `val.jsonl`) plus a
  generated `README.md` dataset card.

Each script defines a constant at the top (`RAW_DIR`, `PROCESSED_DIR`,
`JSONL_DIR`) and uses `Path` objects; expanding the pipeline typically means
changing those values or adding new CLI options.

---

## 🧠 Script Summaries & Patterns

### `01_data_loader.py`
- Loads AI Hub datasets; supports `consultation` and `summary` types via
  `DATASET_CONFIGS`.
- Provides an `AIHubMedicalLoader` class with methods:
  - `explore_structure(path)` for dumping JSON keys/values (run first).
  - `load_consultation_data(...)` returns a list of `{input,output[,category]}`
    dicts and stores it on `self.data`.
  - `get_statistics()` prints pandas statistics.
- `create_sample_data()` generates a small JSON mimicking AI Hub format.
- CLI modes: `--mode sample|load|explore`; the sample path is used to test the
  rest of the pipeline before real data arrives.

### `02_preprocessor.py`
- Text cleaning functions tailored for Korean medical text
  (`normalize_unicode`, `remove_html_tags`, `remove_special_noise`, etc.).
- Quality checks (`is_valid_sample`) filter short/low‑Korean‑ratio/duplicate
  samples.
- `TokenLengthFilter` uses a `transformers.AutoTokenizer` (default
  `beomi/Llama-3-Open-Ko-8B`) to compute token counts; if loading fails, falls
  back to a simple char‑based heuristic.
- `MedicalPreprocessor` orchestrates cleaning, quality filtering, and optional
  token filtering. Stats are printed at the end.
- CLI: `--input`, `--output`, and `--no_token_filter` options.

### `03_jsonl_formatter.py`
- Converts processed records into
  **Alpaca**, **ChatML**, or **LLaMA 3** JSONL formats using
  `FORMAT_FUNCTIONS` and a shared `SYSTEM_PROMPT`.
- `JSONLFormatter` class methods:
  - `convert()` applies formatting to every record.
  - `split_train_val()` shuffles and splits by `train_ratio`.
  - `save_jsonl()` writes one sample per line.
  - `verify_output()` prints a few records for sanity checking.
- A helper `create_dataset_card()` writes a README suitable for HuggingFace Hub.
- CLI: choose format via `--format alpaca|chatml|llama3`.

Common conventions:
- Numeric prefixes enforce pipeline order; do not rename them without adjusting
documentation.
- All significant classes print progress to stdout; new utilities should
follow this pattern rather than silently returning values.
- Korean comments and docstrings are used throughout; maintain that style when
adding features.
- The code avoids external dependencies aside from `pandas` and `transformers`
for preprocessing.

---

## 🛠 Developer Workflows

1. **Exploring new data**: download archive → unpack under `data/raw` → run
   `python 01_data_loader.py --mode explore --file <path>` → inspect output to
   determine correct field names for `input_field`/`output_field`.
2. **Generating sample data**: `python 01_data_loader.py --mode sample`.
3. **Preprocessing**:
   ```bash
   python 02_preprocessor.py --input ./data/raw/<file>.json \
                              --output ./data/processed/clean.json
   ```
   pass `--no_token_filter` if tokenizer setup fails or you want a quick run.
4. **Formatting**:
   ```bash
   python 03_jsonl_formatter.py --input ./data/processed/clean.json \
                                --format llama3 --train_ratio 0.9
   ```
   results appear in `data/jsonl/` along with a README.
5. **Fine‑tuning**: open `04_finetune_T4_colab.ipynb`; follow top instructions
   (T4 GPU, Drive mount, HF token). The notebook contains detailed comments and
   T4‑specific configuration values; modify them only if you know what you're
   doing.

There are no automated tests or build steps; running the CLI scripts is the
primary validation. If you add functionality, mimic the existing argument
parsing and print diagnostic information.

---

## 🔌 External Dependencies & Integration

- **AI Hub**: data must be manually requested and downloaded; scripts assume
  UTF‑8 and JSON/JSONL formats.
- **Transformers**: only used in preprocessing for token counting. Model names
  can be changed but default to a Korean LLaMA‑3 variant.
- **Colab**: training is done in a notebook on Google Colab with T4 GPUs. Drive
  path hard‑coded to `/content/drive/MyDrive/medical_llm`; adjust if you fork
  the repo.
- **HuggingFace Hub**: the notebook logs in using an HF token and can push
  checkpoints. `03_jsonl_formatter` optionally creates a dataset card for upload.

---

## 📌 Project‑Specific Conventions

- Files are prefixed `01_…`, `02_…`, etc., to indicate pipeline order;
  maintain this numbering when adding new stages.
- Use Korean variable names or comments when the domain is medical (e.g.,
  `진료과`, `Q:`/`A:` previews). English is acceptable for generic utilities.
- Scripts should be runnable from the repo root without additional setup.
- When printing paths, resolve against `Path` objects for cross‑platform safety.
- Reuse `json.dump(..., ensure_ascii=False)` to preserve Korean characters.

---

## 🪪 Helpful Examples

- Adding a new dataset type: extend `DATASET_CONFIGS` in
  `01_data_loader.py` with `description`, `expected_format`, `key_fields`, and
  `sample_count`. Follow the existing pattern for automatic list extraction.
- To disable token filtering by default, modify `MedicalPreprocessor.__init__`
  or provide a CLI flag in `02_preprocessor.py`.
- To support a new JSONL format, add a formatter function and register it in
  `FORMAT_FUNCTIONS` in `03_jsonl_formatter.py`.

---

## ✅ Final Notes

There is no existing `.github/copilot-instructions.md`; this file serves as the
source of truth for any AI coding agent. Keep instructions concise and update
whenever the pipeline changes (e.g., if another script is added or the
notebook naming changes).

> **💬 Feedback:** Let me know if any sections are unclear or if more detail is
> needed about a specific script or workflow. We'll iterate accordingly.