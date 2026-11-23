# OpenVINO Embedding Backend for OpenViking

[中文](README.md) | English

OpenViking ships with `bge-small-zh-v1.5` as its default local embedding model: 512 dimensions, 512 token context, CPU inference. That ceiling is too low for a memory system — anything beyond 512 tokens is truncated, so long documents and long session summaries lose a lot of information during extraction and retrieval, and it is unsuitable for a knowledge base.

Moving to a 4096-context, 1024-dimension model fixes the quality problem, but llama.cpp on CPU is far too slow. Measured conditions: a 677M parameter Q8 GGUF model, 22 logical CPU cores, 151.7 token/s throughput, both `n_ctx` and `n_batch` set to 4096.

| Text length | CPU embedding time |
| --- | --- |
| 512 tokens | 2.8 s |
| 2520 tokens | 16.3 s |
| 4000 tokens | 32.2 s |

A document close to the 4096-token limit takes over 30 seconds; rebuilding a vector store in bulk takes over 40 minutes.

This project runs the same int8 model on an Intel integrated GPU through OpenVINO, cutting the same 4096-token embedding from 30.1 seconds to 4.0 seconds. Intel CPU, GPU or NPU can all run int8 quantized models directly; two deployment forms are provided, in-process and HTTP, along with tooling that keeps the memory-extraction model aligned with the host agent.

## Who this is for

| Your situation | Suitable |
| --- | --- |
| Using OpenViking for local memory or a knowledge base on Intel CPU, GPU or NPU | Yes |
| Wanting to escape the llama.cpp performance ceiling on Intel hardware | Yes |
| Needing to switch between in-process and HTTP forms | Yes |
| Using an NVIDIA GPU | llama.cpp's CUDA backend is more direct; OpenVINO offers no advantage |
| Using an AMD GPU | No. OpenVINO support for AMD is limited |
| Only wanting OpenViking's default configuration | Not needed. The built-in bge-small works out of the box |
| Unwilling to re-apply patches after upgrades | Caution. This repository is a patch, not an upstream built-in capability |

The repository has three parts:

| Part | What it solves | Who needs it |
| --- | --- | --- |
| Embedding backend patch | Lets OpenViking support OpenVINO (in-process or HTTP) | Anyone running local embedding on Intel hardware |
| HTTP embedding service | Wraps OpenVINO behind an OpenAI-compatible API | Anyone using the HTTP form or sharing one GPU across instances |
| Memory model sync plugins | Keeps the memory-extraction model aligned with the host system | Anyone running Hermes or DSH |

## Requirements

| Item | Requirement | How to check |
| --- | --- | --- |
| OpenViking | Use the release of this project that matches your version; see that release's notes for the adapted version | `pip show openviking` |
| Python | 3.13 (the venv bundled with OpenViking is fine) | — |
| OpenVINO Runtime | 2026.4 | `python -c "import openvino; print(openvino.__version__)"` |
| Model | OpenVINO IR format (xml, bin, tokenizer) | See the model table below |
| Hardware | Intel CPU works; GPU or NPU needs the matching driver | `python -c "import openvino as ov; print(ov.Core().available_devices)"` |
| Disk | Models from about 600 MB each, compile cache about 2 GB | — |
| Memory | In-process form uses about 1.6 GB resident | Check in Task Manager |

## Platform support

| Platform | Supported | Notes |
| --- | --- | --- |
| Windows 10/11 (x86_64) | Yes | Full support, including Intel integrated GPU acceleration |
| Linux (x86_64) | Yes | Full support; GPU acceleration requires the Intel graphics driver |
| macOS | No | OpenVINO only provides CPU inference on macOS, so GPU acceleration is unavailable |
| ARM (any platform) | No | OpenVINO has no GPU support on ARM |

## Embedding models

All five work out of the box; pick one. They are all int8. The links below use a China-accessible mirror.

| Model | Weights | Dim | Context | Output and pooling | Query prefix | Languages | Download |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 596M | 1024 | 32768 | 3D, last-token | `Instruct: …` | 100+ languages | [Download](https://hf-mirror.com/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) |
| harrier-oss-v1-0.6b | 596M | 1024 | 32768 | 2D, built into model | `Instruct: …` | Multilingual | [Download](https://hf-mirror.com/doldolggi/harrier-oss-v1-0.6b-OpenVINO) |
| bge-m3 | 568M | 1024 | 8192 | 3D, CLS | None | Multilingual | [Download](https://hf-mirror.com/OpenVINO/bge-m3-int8-ov) |
| SEA-LION-ModernBERT-Embedding-600M | 612M | 1024 | 8192 | 3D, mean | None | 11 SEA languages plus Chinese | [Download](https://hf-mirror.com/aisingapore/SEA-LION-ModernBERT-Embedding-600M-OV) |
| embeddinggemma-2 | 744M | **768** | 8192 | 2D, built into model | `task: …` | 100+ languages | [Download](https://hf-mirror.com/wesjos/embeddinggemma-2-openvino-int8) |

The patch registers all five. Choose one and set the matching `model` and `dimension` in `ov.conf`; no manual table entry is needed.

These five are not one model with different weights — the differences must be configured per model, otherwise the vectors come out wrong:

| Difference | Detail |
| --- | --- |
| Dimension | Only embeddinggemma-2 is 768; the other four are 1024. A wrong value fails at startup |
| Output shape | harrier and embeddinggemma-2 already output sentence vectors (2D); the other three output per-token hidden states (3D) and need pooling |
| Pooling | Qwen3 uses last-token, bge-m3 uses CLS, SEA-LION uses mean. A wrong choice noticeably degrades retrieval |
| Normalization | Qwen3, bge-m3 and SEA-LION need explicit normalization, otherwise cosine similarity is distorted; the other two normalize internally |
| Query prefix | harrier and Qwen3 need the `Instruct: …` prefix, bge-m3 requires none, embeddinggemma-2 needs its own on both query and document sides |
| Model file name | The first four use `openvino_model.xml`; embeddinggemma-2's text tower is `text_int8.xml` |

The patch and the HTTP service handle all of this per model. You only need to get `model` and `dimension` right in `ov.conf`.

### Changing models requires a vector store rebuild

A new model may have a different dimension and pooling method, so existing vectors cannot be reused — everything must be re-embedded. This is the only cost of switching models; budget the time.

### Custom models

int8 embedding models are limited in number, so you may want to convert your own. Converted models are outside the patch's built-in set; describe them with `openvino_custom.json` placed next to the model's `tokenizer.json`:

```json
{
  "pooling": "last",
  "normalize": true,
  "query_task": "",
  "query_prefix": "",
  "doc_prefix": ""
}
```

Set `model` to `custom` and `model_path` to the IR file in `ov.conf`. See `models/模型获取.md` for the full field reference and probe method.

## Two interfaces

| | In-process | HTTP service |
| --- | --- | --- |
| Embedding call | OpenVINO Runtime | Standalone HTTP service (8010) |
| Median retrieval latency | 43 ms | 173 to 286 ms |
| Extra process | None | One embedding service process |
| Configuration | `provider: local` | `provider: openai` + `api_base` |
| Best for | Efficient, stable production | Cross-process reuse, shared service across instances |

The HTTP form adds one more layer of request handling, so it is the secondary option.

## Performance

All figures below were measured at 4096 context. Same model `harrier-ov-int8`; the only variable is Intel Arc GPU versus CPU.

| Text length | Note | CPU | GPU | Speedup |
| --- | --- | --- | --- | --- |
| 30 chars | Short query | 55.5 ms | 34.5 ms | 1.6× |
| 1800 chars | Medium | 4255.8 ms | 672.6 ms | 6.3× |
| 6600 chars | Hits the 4096-token limit, truncated | 30149.6 ms | 3985.5 ms | 7.6× |

All three fit within 4096 context; the last one is already at the limit, so longer text does not get slower, it just gets truncated.

End to end: rebuilding a 200-entry memory vector store drops from about 20 minutes to about 3 minutes. The longer the text, the larger the advantage.

### Context length and retrieval quality

A larger context lets more content enter the index intact, but embedding time grows faster. The table below comes from 328 memory files.

The method matches a real store: files are chunked at the given context length, one vector per chunk. Six of the longest files were taken, three questions each — beginning, middle and end — for 18 questions total. For each question we checked whether the top three results contained relevant content; other files on the same topic counted, the target file itself was not required.

| Context | Content fully indexed | Beginning | Middle | End | Total | Chunks | Single 12647-char embed | Rebuild 328 files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 (OpenViking default) | 40.9%, 176 of 298 files truncated | 4/6 (1/6) | 2/6 (1/6) | 3/6 (0/6) | 9/18 | 673 | 2.0 s | 104 s |
| 1024 | 70.5%, 88 truncated | 5/6 (1/6) | 2/6 (1/6) | 4/6 (1/6) | 11/18 | 440 | 4.5 s | 153 s |
| 2048 | 96.3%, 11 truncated | 4/6 (0/6) | 2/6 (1/6) | 5/6 (0/6) | 11/18 | 343 | about 7 s | 323 s |
| 4096 | 100% | 5/6 (5/6) | 4/6 (2/6) | 6/6 (1/6) | 15/18 | 329 | 27 s | 983 s |
| 8192 | 100% | 5/6 (5/6) | 4/6 (2/6) | 6/6 (1/6) | 15/18 | 329 | 67 s | 3434 s |

Reading `4/6 (1/6)`: of 6 questions, 4 had relevant content in the top three results, and in 1 of those the target file itself was among them.

A few notes:

- **At 4096, 15 of 18 questions found relevant content**, 8 of them hitting the target file directly. At 512 only 9 found anything — a clear gap.
- **Beginning content is found at every level** (4 to 5 times). The opening chunk survives at any context length, so low context does not hurt beginning retrieval.
- **The middle is where low context loses most.** Levels 512 through 2048 found middle content only twice; 4096 found it 4 times. Middle detail simply does not enter the index at low context.
- **The end improves clearly from 2048** (3 to 5 times) because 96% of files are fully indexed; at 4096 all 6 were found.
- **One easily misread point.** At low context the target file itself rarely reaches the top three (only 3 times at 512), but the top three are often other files on the same topic, so the user does read relevant content. Ranking by the target file alone underestimates quality, which is why the table above counts "relevant content" and lists the target-file count in parentheses.
- **8192 and 4096 give identical results.** The longest file in the store is 3893 tokens, all within 4096, so 8192 covers nothing extra while costing 3.5× the build time. 4096 is recommended.
- Six files and 18 questions per level is a small sample; treat it as directional, not statistical.

Going from 4096 to 8192 raises a single long embedding from 3.9 to 12.1 seconds, 3.1×, but it runs in the background. The current store's longest file is 3893 tokens, already covered by 4096.

## Keeping the memory-extraction model in sync

OpenViking's memory extraction calls an LLM to generate summaries, with the endpoint hand-maintained and no configuration option. It easily drifts from the host system: when the host switches or retires a model, OpenViking's config does not follow, extraction starts failing, and the failure shows up only in logs.

This project ships custom plugins for Hermes and DSH that sync the model automatically, writing the model actually in use into OpenViking's config:

| Plugin | Host | Language |
| --- | --- | --- |
| `hermes-vlm-follow` | Hermes | Python |
| `dsh-vlm-follow` | DSH | JavaScript |

OpenViking rereads its config every 30 seconds, so changes take effect within half a minute without restarting the service.

## One-command install

The project ships an installer (Chinese and English), handling environment probing, config writing, patching, plugin installation and verification.

### Entry points

| System | Entry |
| --- | --- |
| Windows | Double-click `install.cmd`, or run `install.ps1` |
| Linux | `python install.py` |

### What the wizard does

| Step | Content |
| --- | --- |
| 1 Probe | Checks OpenViking, OpenVINO Runtime, the patch target and agent-side plugins; lists available models |
| 2 Embedding form | In-process or HTTP |
| 3 Model | One of the five built-in models, or your own |
| 4 Agents | Which agents connected to OpenViking should get the sync plugin |
| 5 Confirm | Lists every action it will take, with target paths |
| 6 Execute | Backs up before each change; any failure rolls everything back |
| 7 Finish | Prints the restart command and what the startup log should show |

OpenVINO Runtime and tokenizers are installed automatically when missing; you do not need to prepare them.

### Common options

| Option | Effect |
| --- | --- |
| `--dry-run` | Runs all probing and prompts, prints what it would do, changes nothing |
| `--ov-home <path>` | Points at the OpenViking installation, skipping auto-detection |
| `--lang <auto\|zh\|en>` | Interface language, automatic by default |
| `--version` | Shows the version |

If more than one OpenViking is installed, use `--ov-home` to say which one to target.

### First run

Run a dry run first to confirm the paths, models and agents it detects:

```bash
python install.py --dry-run --ov-home /path/to/OpenViking
```

### Prerequisites

The wizard only detects and informs; it does not install these for you:

| Item | Requirement |
| --- | --- |
| OpenViking | Official release; the exact version is stated in the release notes of the build you use |
| Agent-side OpenViking plugin | At least one agent connected to OpenViking and working |
| Embedding model | One of the five built-ins downloaded to the cache directory, or your own OpenVINO IR model |

How agent-side plugins are detected: for Hermes, whether `plugins/openviking/` exists; for DSH, whether the profile's `package.json` depends on `@openviking/`.

If agents are installed but none is connected to OpenViking, the wizard stops at the probe stage and lists what each one is missing.

## Quick start: in-process

1. Prepare an OpenVINO IR model and place it in the embedding model directory.
2. Apply the backend patch:

```bash
python patches/patch_openvino.py --check    # inspect first
python patches/patch_openvino.py            # apply
```

The patch is adapted against the official release and applies directly, with no prior modification needed. `--check` reports whether this patch is already applied:

| `--check` state | Meaning | Exit code |
| --- | --- | --- |
| Not patched | Not applied yet | 2 |
| Patched | Already current | 0 |

The script is safe to run repeatedly. If the target file's structure does not match the adapted version, it stops with an error and writes nothing.

3. Copy `configs/ov.conf.local.example` to `ov.conf` and fill in the model path and credentials.
4. Restart the OpenViking service. The startup log should show `OpenVINO embedder ready: model=... device=GPU`.

## Quick start: HTTP

1. Start the embedding service, naming the model preset with `--model`:

```bash
python http-server/embed_server.py --model harrier-ov-int8 --model-dir D:/OpenViking/models/harrier-ov-int8 --port 8010
```

`--model` accepts the same five values as the table above. Each model places its IR differently; the service finds it by preset: `openvino_model.xml` for the first four (harrier inside `int8/`), `text_int8.xml` for embeddinggemma-2.

2. Apply the keepalive patch (without it, bulk writes reconnect constantly):

```bash
python patches/patch_http_keepalive.py
```

3. Build `ov.conf` from `configs/ov.conf.http.example`.
4. Restart OpenViking and confirm `/health` and the embedding endpoint are reachable.

## Re-apply patches after upgrading

`pip install --upgrade openviking` replaces the entire `site-packages/openviking/` tree, discarding both patches. Re-run them in order:

```bash
python patches/patch_openvino.py
python patches/patch_http_keepalive.py    # HTTP form only
```

You can also re-run the installer; it detects that the patch is gone and reapplies it.

## License

MIT. The patch scripts and HTTP service are original to this repository; OpenViking itself follows its upstream license.

## Upstream

This repository is a third-party extension of OpenViking and is not affiliated with ByteDance. If upstream adds a built-in OpenVINO backend, this project will note the corresponding version and move into maintenance.
