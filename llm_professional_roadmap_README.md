# LLM Technical Roadmap

A professional roadmap for building deep technical competence in Large Language Models, LLM systems, RAG, agents, inference, evaluation, and production architecture.

This roadmap is not a schedule. No timing is attached. It is a dependency map: each layer builds on the previous one and produces concrete artifacts that prove competence.

---

## Table of Contents

1. [Target Outcome](#target-outcome)
2. [How to Use This Roadmap](#how-to-use-this-roadmap)
3. [Competency Ladder](#competency-ladder)
4. [Core Roadmap Layers](#core-roadmap-layers)
   - [Layer 1 — LLM Foundations](#layer-1--llm-foundations)
   - [Layer 2 — Training Pipeline](#layer-2--training-pipeline)
   - [Layer 3 — Post-Training](#layer-3--post-training)
   - [Layer 4 — Reasoning Models](#layer-4--reasoning-models)
   - [Layer 5 — Inference Fundamentals](#layer-5--inference-fundamentals)
   - [Layer 6 — Serving Engines](#layer-6--serving-engines)
   - [Layer 7 — KV Cache and Long Context](#layer-7--kv-cache-and-long-context)
   - [Layer 8 — Quantization and Compression](#layer-8--quantization-and-compression)
   - [Layer 9 — RAG Systems](#layer-9--rag-systems)
   - [Layer 10 — Agentic Systems](#layer-10--agentic-systems)
   - [Layer 11 — Evaluation and Benchmarking](#layer-11--evaluation-and-benchmarking)
   - [Layer 12 — Production Architecture](#layer-12--production-architecture)
5. [Advanced Tracks](#advanced-tracks)
6. [Master Portfolio](#master-portfolio)
7. [Decision Rules](#decision-rules)
8. [Capstone Projects](#capstone-projects)
9. [Reference Sources](#reference-sources)
10. [Final Mental Model](#final-mental-model)

---

## Target Outcome

The target is not to become someone who follows model releases.

The target is this:

```text
Given a real domain problem,
you can select the model,
design the system,
choose the serving stack,
build the retrieval or agent layer,
measure quality,
control latency and cost,
debug failures,
and explain every tradeoff technically.
```

A serious LLM engineer must connect seven layers:

```text
model internals
+ training recipe
+ post-training behavior
+ inference runtime
+ retrieval / agent architecture
+ evaluation discipline
+ production constraints
```

If one layer is missing, the system becomes fragile.

---

## How to Use This Roadmap

Use this roadmap as a technical progression, not a reading list.

For every layer, produce three things:

1. **Notes** — concise explanations in your own words.
2. **Code** — a minimal implementation or experiment.
3. **Evidence** — benchmark results, eval reports, dashboards, traces, or architecture diagrams.

Do not move by passive reading.

Move only when you can answer:

```text
What mechanism did I understand?
What failure mode can I now diagnose?
What artifact proves it?
What decision can I now make better?
```

---

## Competency Ladder

### Level 1 — API User

Can call hosted LLM APIs and write prompts.

Insufficient for serious engineering.

### Level 2 — Prototype Builder

Can build simple chatbots, RAG demos, and agent demos.

Useful, but fragile.

### Level 3 — LLM Engineer

Can build, evaluate, debug, and deploy LLM systems with controlled quality.

Minimum professional target.

### Level 4 — LLM Systems Engineer

Can optimize inference, latency, throughput, GPU memory, cost, batching, routing, and observability.

Strong industry level.

### Level 5 — Research Engineer

Can read papers mechanically, reproduce core ideas, modify training/post-training/inference methods, and evaluate tradeoffs.

Deep technical level.

### Level 6 — LLM Architect

Can design organization-scale LLM platforms with governance, multi-tenancy, security, cost controls, evaluation gates, and production reliability.

Architecture-level target.

---

# Core Roadmap Layers

---

## Layer 1 — LLM Foundations

### Goal

Understand the mechanical path from raw text to generated token.

### Core concepts

- tokens
- token IDs
- embeddings
- hidden states
- decoder-only Transformers
- self-attention
- Q/K/V projections
- causal masking
- multi-head attention
- grouped-query attention
- multi-query attention
- MLP blocks
- activation functions
- RMSNorm / LayerNorm
- residual connections
- RoPE and positional encoding
- logits
- softmax
- decoding
- temperature
- top-k and top-p sampling
- KV cache
- prefill vs decode

### Required understanding

You must be able to explain this pipeline:

```text
text
→ tokenizer
→ token IDs
→ embeddings
→ Transformer blocks
→ final hidden state
→ logits
→ probability distribution
→ decoding
→ next token
→ updated context
```

### Deep technical questions

- Why does the model predict tokens instead of words?
- Why does tokenization affect cost, latency, and multilingual quality?
- Why does attention need Q, K, and V?
- Why is causal masking required for decoder-only models?
- Why does GQA reduce KV cache memory?
- Why does generation happen sequentially?
- Why does the model output probabilities rather than facts?
- Why can fluent text still be false?

### Practical artifacts

Build:

```text
01_tokenizer_inspection.ipynb
02_tiny_decoder_transformer.py
03_sampling_playground.ipynb
04_kv_cache_toy_decoder.py
```

### Evaluation gate

You pass this layer when you can explain one generated token from text input to output probability without saying vague phrases like:

```text
The model understands the sentence.
```

Use mechanical language instead:

```text
The model maps token IDs to vectors, routes information through attention, transforms representations through MLPs, projects the final hidden state into logits, then samples from a distribution.
```

---

## Layer 2 — Training Pipeline

### Goal

Understand how base models acquire broad capability before alignment.

### Core concepts

- corpus construction
- data filtering
- deduplication
- language balancing
- domain mixtures
- tokenizer training
- next-token objective
- cross-entropy loss
- perplexity
- scaling laws
- compute-optimal training
- batch size
- gradient accumulation
- optimizer behavior
- AdamW
- learning rate schedules
- warmup
- checkpointing
- distributed training
- data contamination
- benchmark leakage

### Required understanding

Base model quality is mostly determined before post-training.

The training pipeline is not just:

```text
collect data → train model
```

It is closer to:

```text
source selection
→ cleaning
→ filtering
→ deduplication
→ mixture design
→ tokenizer training
→ architecture selection
→ compute planning
→ optimization
→ checkpoint evaluation
→ contamination checks
```

### Deep technical questions

- Why can low-quality data damage a large model more than lack of data?
- Why is deduplication important?
- Why is benchmark contamination hard to detect?
- Why does tokenizer design affect Arabic, code, and math differently?
- What does loss measure?
- Why is lower loss not always better downstream behavior?
- What does compute-optimal training mean?
- Why does overtraining on too few tokens waste compute?

### Practical artifacts

Build:

```text
01_data_cleaning_pipeline.py
02_bpe_tokenizer_training.ipynb
03_tiny_pretraining_run.py
04_loss_curve_analysis.md
05_contamination_checklist.md
```

### Evaluation gate

You pass this layer when you can inspect a model technical report and identify:

```text
training data quality
training token count
architecture size
compute budget
loss behavior
benchmark contamination risk
missing ablations
```

---

## Layer 3 — Post-Training

### Goal

Understand how base models become useful, instruction-following assistants.

### Core concepts

- supervised fine-tuning
- instruction tuning
- chat templates
- preference data
- RLHF
- RLAIF
- reward models
- DPO
- IPO
- KTO
- ORPO
- GRPO
- RLVR
- rejection sampling
- Constitutional AI
- safety tuning
- refusal shaping
- tone and style tuning
- reasoning distillation

### Required understanding

Pretraining teaches broad statistical capability.

Post-training shapes behavior.

The transition is:

```text
base completion model
→ instruction-following model
→ preference-optimized assistant
→ safety-shaped assistant
→ domain-adapted assistant
```

### Deep technical questions

- What does SFT teach that pretraining does not?
- Why can SFT make a model more obedient but less general?
- Why does preference optimization change behavior without adding new knowledge reliably?
- What is the difference between a reward model and a preference objective?
- Why can alignment improve helpfulness while damaging calibration?
- When should you fine-tune versus use RAG?
- When should you use LoRA instead of full fine-tuning?

### Practical artifacts

Build:

```text
01_instruction_dataset_schema.jsonl
02_sft_training_run.py
03_preference_pair_dataset.jsonl
04_dpo_training_run.py
05_before_after_eval_report.md
```

### Evaluation gate

You pass this layer when you can explain why this sentence is usually incomplete:

```text
We should fine-tune the model.
```

A complete answer asks:

```text
What behavior are we changing?
Is the missing capability knowledge, instruction following, format compliance, preference, safety, or domain style?
Can retrieval solve it instead?
Do we have evals to prove improvement?
```

---

## Layer 4 — Reasoning Models

### Goal

Understand models that allocate more inference-time computation to solve harder tasks.

### Core concepts

- chain-of-thought
- hidden reasoning traces
- reasoning tokens
- self-consistency
- majority voting
- verifier models
- outcome supervision
- process supervision
- reward models for reasoning
- RL with verifiable rewards
- test-time compute
- search over reasoning paths
- reasoning distillation
- concise reasoning
- overthinking
- thinking/non-thinking modes

### Required understanding

Reasoning models are not magic.

They combine three ideas:

```text
more intermediate computation
+ better reward signals
+ stronger verification or selection
```

The model is allowed to spend more tokens or internal computation before answering.

This can improve math, code, logic, and planning tasks.

It can also increase latency, cost, verbosity, and false confidence.

### Deep technical questions

- When does longer reasoning improve accuracy?
- When does it create noise?
- Why are math and code easier for RLVR than open-ended strategy?
- What is the difference between outcome supervision and process supervision?
- Why can a model produce a plausible reasoning trace for a wrong answer?
- When should a tool replace reasoning?

### Practical artifacts

Build:

```text
01_reasoning_eval_dataset.jsonl
02_baseline_vs_reasoning_eval.ipynb
03_verifier_experiment.py
04_majority_vote_experiment.py
05_reasoning_failure_taxonomy.md
```

### Evaluation gate

You pass this layer when you can decide:

```text
normal instruct model
vs reasoning model
vs retrieval
vs external tool
vs deterministic solver
```

based on task type, latency, verifiability, and cost.

---

## Layer 5 — Inference Fundamentals

### Goal

Understand LLM inference as a systems problem.

### Core concepts

- request lifecycle
- model loading
- tokenizer overhead
- prefill
- decode
- TTFT
- TPOT
- tokens per second
- latency percentiles
- batch size
- dynamic batching
- continuous batching
- streaming
- chunked prefill
- scheduling
- memory bandwidth
- compute-bound vs memory-bound phases
- GPU utilization
- CUDA graphs
- FlashAttention
- speculative decoding

### Required understanding

Inference has two different workloads:

```text
Prefill: process prompt tokens.
Decode: generate output tokens one by one.
```

Prefill is usually parallel and compute-heavy.

Decode is sequential and often memory-bandwidth-bound.

This explains why one metric is not enough.

### Deep technical questions

- Why does long input increase TTFT?
- Why does long output increase total latency?
- Why does throughput improve with batching but latency can degrade?
- Why are p95 and p99 latency more important than average latency?
- Why does decode often underutilize GPU compute?
- Why does streaming improve perceived latency but not total compute cost?

### Practical artifacts

Build:

```text
01_inference_benchmark_runner.py
02_ttft_tpot_dashboard.ipynb
03_batch_size_experiment.md
04_context_length_experiment.md
05_streaming_vs_non_streaming_report.md
```

Track:

```text
TTFT
TPOT
tokens/sec
p50 latency
p95 latency
p99 latency
VRAM
GPU utilization
concurrency
error rate
```

### Evaluation gate

You pass this layer when you stop saying:

```text
This model is fast.
```

and instead say:

```text
At this context length, batch size, concurrency, hardware, precision, and serving engine, the system achieves this TTFT, TPOT, throughput, and p95 latency.
```

---

## Layer 6 — Serving Engines

### Goal

Choose and operate the correct inference runtime.

### Core concepts

- vLLM
- SGLang
- TensorRT-LLM
- llama.cpp
- Ollama
- Hugging Face TGI
- LMDeploy
- exllama
- OpenAI-compatible API servers
- model loading
- quantization support
- tensor parallelism
- pipeline parallelism
- expert parallelism
- LoRA serving
- structured outputs
- prefix caching
- speculative decoding
- prefill/decode disaggregation

### Required understanding

A serving engine is not a wrapper.

It controls:

```text
memory layout
KV cache allocation
request scheduling
batching
kernel execution
parallelism
quantization support
structured decoding
streaming
observability hooks
```

### Engine selection map

| Engine | Best use case | Main strength | Main caution |
|---|---|---|---|
| vLLM | General production serving | High-throughput serving, PagedAttention, batching | Model/feature compatibility must be checked |
| SGLang | Advanced serving workflows | Prefix caching, structured outputs, strong runtime features | Fast-moving stack; validate deployment stability |
| TensorRT-LLM | NVIDIA-optimized production | High performance on NVIDIA GPUs | Higher complexity and build/deploy friction |
| llama.cpp | Local CPU/GPU inference | GGUF ecosystem, local experimentation | Not ideal as a high-concurrency production server |
| Ollama | Developer convenience | Easy local model management | Not a full production serving architecture |
| TGI | HF ecosystem serving | Straightforward deployment | Compare performance against vLLM/SGLang per workload |

### Deep technical questions

- Does the engine support the target model architecture?
- Does it support the required quantization format?
- Does it support the needed context length?
- Does it support continuous batching?
- Does it support prefix caching?
- Does it support structured outputs?
- Does it support LoRA adapters?
- Can it scale across GPUs?
- Can it expose useful metrics?

### Practical artifacts

Build:

```text
01_serving_engine_comparison.md
02_vllm_deployment_config.yaml
03_sglang_deployment_config.yaml
04_tensorrt_llm_notes.md
05_runtime_decision_matrix.xlsx
```

### Evaluation gate

You pass this layer when you can defend runtime choice using workload constraints instead of preference.

---

## Layer 7 — KV Cache and Long Context

### Goal

Understand the memory system behind long-context inference.

### Core concepts

- KV cache
- layers
- KV heads
- head dimension
- precision
- context length
- batch size
- cache paging
- PagedAttention
- prefix caching
- cache reuse
- cache eviction
- cache offloading
- KV cache quantization
- sliding window attention
- attention sinks
- RoPE scaling
- YaRN
- NTK scaling
- lost-in-the-middle
- context pollution

### Required understanding

Long context is not just an architecture feature.

It is a serving constraint.

KV cache memory grows with:

```text
batch size × context length × layers × KV heads × head dimension × bytes per value
```

Because both K and V are stored, the multiplier includes both tensors.

### Deep technical questions

- Why can the weights fit but serving still fail?
- Why does long context reduce concurrency?
- Why does GQA reduce KV cache memory?
- Why does prefix caching help chat and agents?
- Why does long context not guarantee long-context reasoning?
- Why do models lose information in the middle?
- When should you use RAG instead of increasing context?

### Practical artifacts

Build:

```text
01_kv_cache_calculator.py
02_context_scaling_benchmark.ipynb
03_prefix_cache_experiment.md
04_long_context_eval_set.jsonl
05_lost_in_the_middle_report.md
```

### Evaluation gate

You pass this layer when you can estimate whether a context/concurrency target fits in GPU memory before running the workload.

---

## Layer 8 — Quantization and Compression

### Goal

Reduce memory and cost without silently destroying quality.

### Core concepts

- FP32
- FP16
- BF16
- FP8
- INT8
- INT4
- weight-only quantization
- activation quantization
- KV cache quantization
- per-tensor quantization
- per-channel quantization
- group-wise quantization
- symmetric vs asymmetric quantization
- calibration data
- GPTQ
- AWQ
- SmoothQuant
- GGUF
- bitsandbytes
- QLoRA
- LoRA merging
- pruning
- distillation

### Required understanding

Quantization is controlled numerical damage.

The question is not:

```text
Can the model run in 4-bit?
```

The question is:

```text
Does this quantized model preserve the required behavior under the target workload, domain, serving engine, hardware, context length, and eval set?
```

### Deep technical questions

- Are weights, activations, or KV cache being quantized?
- Is the quantization format supported efficiently by the serving engine?
- What calibration data was used?
- Does quality degrade globally or only on hard examples?
- Does quantization damage reasoning more than simple Q&A?
- Does it break JSON formatting or tool calls?
- Does it improve latency or only memory footprint?

### Practical artifacts

Build:

```text
01_quantization_matrix.md
02_fp16_vs_int8_vs_int4_benchmark.ipynb
03_awq_gptq_comparison.md
04_kv_quantization_experiment.md
05_quantization_failure_report.md
```

Track:

```text
VRAM
TTFT
TPOT
throughput
accuracy
format validity
RAG faithfulness
code correctness
reasoning accuracy
```

### Evaluation gate

You pass this layer when you reject any claim like:

```text
4-bit is fine.
```

unless it comes with domain-specific eval results.

---

## Layer 9 — RAG Systems

### Goal

Build retrieval systems that improve factuality instead of adding noisy context.

### Core concepts

- ingestion
- parsing
- OCR
- layout extraction
- chunking
- semantic chunking
- metadata
- embeddings
- vector databases
- BM25
- dense retrieval
- hybrid retrieval
- Reciprocal Rank Fusion
- reranking
- cross-encoders
- query rewriting
- multi-query retrieval
- parent-child chunks
- contextual compression
- citation grounding
- freshness
- permissions
- GraphRAG
- retrieval evaluation
- answer evaluation

### Required understanding

RAG is not:

```text
embed documents → retrieve chunks → put into prompt
```

Production RAG is:

```text
parse documents
→ preserve structure
→ chunk with metadata
→ index lexical and semantic signals
→ retrieve candidates
→ fuse results
→ rerank
→ assemble context
→ generate grounded answer
→ verify citations
→ evaluate retriever and generator separately
```

### Deep technical questions

- Did the retriever find the right evidence?
- Did the reranker improve or damage ranking?
- Did the generator use the evidence?
- Are citations actually supporting the answer?
- Is the chunk too small, too large, or badly split?
- Is the failure lexical, semantic, temporal, or permission-related?
- Would GraphRAG help or overcomplicate the system?

### Practical artifacts

Build:

```text
01_document_ingestion_pipeline.py
02_hybrid_retriever.py
03_reranking_pipeline.py
04_rag_eval_dataset.jsonl
05_citation_verifier.py
06_rag_failure_dashboard.ipynb
```

Track:

```text
hit rate
MRR
nDCG
context precision
context recall
faithfulness
answer relevance
citation support
latency
cost
```

### Evaluation gate

You pass this layer when you can debug a wrong RAG answer by locating the failing stage:

```text
parsing
chunking
embedding
retrieval
fusion
reranking
context assembly
generation
citation verification
```

---

## Layer 10 — Agentic Systems

### Goal

Build tool-using systems with bounded control, not uncontrolled loops.

### Core concepts

- tool calling
- function schemas
- structured outputs
- ReAct
- planners
- routers
- state machines
- graph workflows
- memory
- scratchpads
- retries
- validators
- verifiers
- multi-agent systems
- human approval
- tool permissions
- sandboxing
- budget limits
- observability
- traces
- agent evaluation

### Required understanding

Most production agents should start as workflows.

A safe design is:

```text
explicit state
+ bounded tools
+ schema validation
+ retry limits
+ budget limits
+ human approval for risky actions
+ trace logging
+ evaluation
```

### Deep technical questions

- Is this task actually agentic?
- Can a deterministic workflow solve it better?
- Which state must persist?
- Which tools can cause damage?
- Which actions require approval?
- How does the system stop loops?
- How are tool outputs validated?
- How are partial failures handled?

### Practical artifacts

Build:

```text
01_tool_schema_registry.json
02_bounded_agent_workflow.py
03_router_policy.md
04_agent_trace_viewer.ipynb
05_agent_eval_dataset.jsonl
06_agent_failure_taxonomy.md
```

### Evaluation gate

You pass this layer when you can explain why this is dangerous:

```text
Let the agent decide what to do.
```

and replace it with:

```text
Define allowed actions, state transitions, approval gates, budgets, and failure handling.
```

---

## Layer 11 — Evaluation and Benchmarking

### Goal

Measure LLM systems before production users discover failure.

### Core concepts

- golden datasets
- synthetic evals
- human evals
- LLM-as-judge
- pairwise comparison
- rubrics
- evaluator calibration
- retrieval metrics
- generation metrics
- faithfulness
- answer relevance
- context precision
- context recall
- format validity
- safety evals
- latency evals
- cost evals
- regression tests
- drift monitoring
- online A/B tests
- production traces

### Required understanding

Evaluation must be component-specific.

A full system eval separates:

```text
retrieval quality
generation quality
format compliance
factual grounding
latency
cost
safety
user-impact metrics
```

A single “accuracy” number is usually insufficient.

### Deep technical questions

- What exactly is being measured?
- Is the eval set representative?
- Are hard negatives included?
- Is the judge model calibrated?
- Does the rubric punish unsupported claims?
- Does the eval detect regressions?
- Does quality hold under production latency constraints?
- Do evals include adversarial and edge cases?

### Practical artifacts

Build:

```text
01_eval_dataset_schema.json
02_rag_eval_harness.py
03_llm_judge_rubric.md
04_human_review_protocol.md
05_regression_test_suite.py
06_eval_dashboard.ipynb
```

Track:

```text
quality score
faithfulness
citation support
retrieval recall
format validity
refusal correctness
latency
cost per request
failure category
regression delta
```

### Evaluation gate

You pass this layer when every LLM system you ship has a release gate:

```text
minimum quality
maximum hallucination rate
maximum p95 latency
maximum cost
minimum citation support
required safety behavior
regression threshold
```

---

## Layer 12 — Production Architecture

### Goal

Design deployable, observable, secure, cost-controlled LLM systems.

### Core concepts

- API gateway
- auth
- tenant isolation
- model gateway
- prompt builder
- model routing
- fallback models
- cache layers
- RAG service
- tool service
- serving engine
- GPU scheduling
- Kubernetes
- autoscaling
- rate limiting
- quota management
- observability
- tracing
- audit logs
- cost accounting
- secrets management
- PII handling
- access control
- prompt injection defense
- data retention
- disaster recovery
- SLA design

### Required understanding

Production architecture is about failure control.

The design must answer:

```text
What happens when the model is slow?
What happens when retrieval fails?
What happens when a tool returns bad data?
What happens when the user attempts prompt injection?
What happens when the GPU is full?
What happens when cost spikes?
What happens when the model changes behavior?
```

### Deep technical questions

- Where is tenant isolation enforced?
- Where are prompts logged?
- Where is PII removed or protected?
- How is cost attributed?
- How are traces connected across retrieval, tools, and model calls?
- What is cached?
- What must never be cached?
- How are risky tool calls approved?
- How are eval results connected to deployment decisions?

### Practical artifacts

Build:

```text
01_reference_architecture_diagram.png
02_model_gateway_spec.md
03_rag_service_spec.md
04_tool_execution_policy.md
05_observability_schema.json
06_cost_model.xlsx
07_security_threat_model.md
08_production_readiness_checklist.md
```

### Evaluation gate

You pass this layer when you can write an architecture review covering:

```text
quality
latency
cost
scaling
security
privacy
evaluation
monitoring
failure modes
rollback plan
```

---

# Advanced Tracks

---

## Advanced Track A — Multimodal LLMs

### Scope

- vision-language models
- image encoders
- audio encoders
- video frame sampling
- document understanding
- OCR
- layout-aware parsing
- visual grounding
- multimodal RAG
- multimodal evals

### Artifact

```text
multimodal_document_qa_system/
├── ocr_pipeline.py
├── layout_parser.py
├── image_captioner.py
├── multimodal_index.py
├── answer_generator.py
└── eval_report.md
```

### Gate

You should know when OCR + text RAG is sufficient and when a true vision-language model is required.

---

## Advanced Track B — Domain Adaptation

### Scope

- prompting
- RAG
- LoRA
- QLoRA
- SFT
- continued pretraining
- domain terminology
- ontology grounding
- legal/medical/financial evals
- private data governance

### Decision hierarchy

```text
Prompting first
RAG second
LoRA/SFT third
continued pretraining only when clearly justified
```

### Artifact

```text
domain_adaptation_decision_doc.md
```

Must include:

```text
problem
missing capability
available data
privacy constraints
freshness needs
latency target
recommended method
eval plan
risk
```

### Gate

You should be able to reject unnecessary fine-tuning.

---

## Advanced Track C — LLM Security

### Scope

- prompt injection
- indirect prompt injection
- jailbreaks
- retrieval poisoning
- tool abuse
- data exfiltration
- unsafe code execution
- PII leakage
- over-permissive agents
- sandboxing
- allowlists
- audit logging

### Artifact

```text
llm_red_team_suite/
├── prompt_injection_cases.jsonl
├── retrieval_poisoning_cases.jsonl
├── tool_abuse_cases.jsonl
├── pii_leakage_cases.jsonl
├── jailbreak_cases.jsonl
└── mitigation_report.md
```

### Gate

You should treat safety as architecture, not prompt wording.

---

## Advanced Track D — Hardware-Aware LLM Engineering

### Scope

- GPU memory hierarchy
- HBM bandwidth
- tensor cores
- CUDA kernels
- FlashAttention
- NCCL
- NVLink
- PCIe bottlenecks
- tensor parallelism
- pipeline parallelism
- expert parallelism
- CPU offload
- quantized kernels
- serving topology

### Artifact

```text
hardware_fit_sheet.md
```

Must include:

```text
model
precision
weights memory
KV cache memory
context target
concurrency target
GPU type
interconnect
serving engine
expected bottleneck
risk
```

### Gate

You should know whether the bottleneck is compute, memory bandwidth, network, scheduling, or storage.

---

## Advanced Track E — Research Literacy

### Scope

- ablation analysis
- benchmark selection
- contamination checks
- compute budget
- data mixture
- architecture delta
- training recipe
- inference-time tricks
- statistical significance
- reproducibility
- production implication

### Paper reading template

```text
Title:
Claim:
Mechanism:
What changed:
What stayed constant:
Dataset:
Compute:
Ablation:
Metric:
Weakness:
Can I reproduce it:
Production implication:
```

### Gate

You should summarize papers by mechanism, not by headline.

---

# Master Portfolio

Build this portfolio to prove practical depth.

```text
llm-technical-portfolio/
├── 01_tiny_transformer/
├── 02_tokenizer_comparison/
├── 03_pretraining_pipeline/
├── 04_sft_dpo_experiment/
├── 05_reasoning_eval_harness/
├── 06_inference_benchmark_suite/
├── 07_serving_engine_comparison/
├── 08_kv_cache_calculator/
├── 09_quantization_benchmark/
├── 10_rag_system/
├── 11_agentic_workflow/
├── 12_eval_dashboard/
├── 13_production_architecture/
├── 14_security_red_team_suite/
├── 15_cost_model/
└── 16_paper_review_database/
```

Each artifact should include:

```text
README.md
setup instructions
experiment script
sample data or data schema
results
failure modes
decision rules
```

---

# Decision Rules

## Model selection

```text
Choose the smallest model that satisfies quality, latency, cost, and safety requirements under eval.
```

## Fine-tuning

```text
Fine-tune only when the missing behavior cannot be solved cleanly with prompting, retrieval, routing, or tools.
```

## RAG

```text
Use RAG when knowledge must be fresh, private, cited, inspectable, or too large to fit reliably in model weights.
```

## Agents

```text
Use agents only when the task needs multi-step tool use under uncertainty. Use workflows when the process is known.
```

## Reasoning models

```text
Use reasoning models when the task benefits from extra inference-time computation and the answer can be verified or judged reliably.
```

## Quantization

```text
Quantize only against a domain eval set. Memory reduction without quality measurement is not optimization.
```

## Long context

```text
Long context is not a replacement for retrieval, ranking, summarization, or context design.
```

## Serving

```text
Report TTFT, TPOT, p95 latency, throughput, context length, concurrency, hardware, precision, and serving engine together.
```

## Evaluation

```text
No eval set means no controlled improvement.
```

## Production

```text
Production LLM systems require architecture-level controls: auth, logging, tracing, quotas, safety, rollback, monitoring, and cost accounting.
```

---

# Capstone Projects

## Capstone 1 — Local LLM Inference Lab

Build a benchmark environment comparing multiple serving engines.

Required components:

```text
vLLM or SGLang server
local model
benchmark runner
TTFT/TPOT tracker
VRAM tracker
concurrency test
context length test
results report
```

Success condition:

```text
You can recommend a serving engine for a defined workload with measured evidence.
```

---

## Capstone 2 — Enterprise RAG System

Build a RAG system over realistic documents.

Required components:

```text
ingestion
chunking
metadata
BM25
dense retrieval
hybrid retrieval
reranking
citations
eval dataset
dashboard
failure taxonomy
```

Success condition:

```text
You can separate retrieval failure from generation failure and improve each independently.
```

---

## Capstone 3 — Bounded Agent Workflow

Build an agent that uses tools under strict control.

Required components:

```text
state machine
tool schemas
router
validator
retry policy
budget limit
approval gate
trace logs
eval set
```

Success condition:

```text
The system completes useful multi-step tasks without uncontrolled autonomy.
```

---

## Capstone 4 — LLM Evaluation Platform

Build a reusable eval harness.

Required components:

```text
golden dataset
LLM judge rubric
human review protocol
retrieval metrics
generation metrics
latency metrics
cost metrics
regression gates
report generator
```

Success condition:

```text
Every model, prompt, retriever, and agent change can be compared against a baseline.
```

---

## Capstone 5 — Production LLM Architecture Review

Design a production-grade system for a real organization.

Required components:

```text
architecture diagram
model gateway
RAG service
tool service
serving layer
monitoring
security model
cost model
data retention plan
rollback plan
risk register
```

Success condition:

```text
The architecture can survive review by engineering, security, compliance, and business stakeholders.
```

---

# Reference Sources

Use primary sources first.

## Foundations

- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- GPT-3: Language Models are Few-Shot Learners — https://arxiv.org/abs/2005.14165
- RoFormer / RoPE — https://arxiv.org/abs/2104.09864
- RMSNorm — https://arxiv.org/abs/1910.07467
- GLU Variants Improve Transformer — https://arxiv.org/abs/2002.05202
- Grouped-Query Attention — https://arxiv.org/abs/2305.13245

## Training and scaling

- Scaling Laws for Neural Language Models — https://arxiv.org/abs/2001.08361
- Training Compute-Optimal Large Language Models — https://arxiv.org/abs/2203.15556
- LLaMA paper — https://arxiv.org/abs/2302.13971
- The Pile — https://arxiv.org/abs/2101.00027

## Post-training and reasoning

- InstructGPT — https://arxiv.org/abs/2203.02155
- Constitutional AI — https://arxiv.org/abs/2212.08073
- Direct Preference Optimization — https://arxiv.org/abs/2305.18290
- DeepSeek-R1 — https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report — https://arxiv.org/abs/2505.09388

## Inference and serving

- vLLM documentation — https://docs.vllm.ai/
- vLLM PagedAttention — https://docs.vllm.ai/en/latest/design/paged_attention/
- SGLang documentation — https://docs.sglang.io/
- TensorRT-LLM documentation — https://docs.nvidia.com/tensorrt-llm/index.html
- llama.cpp — https://github.com/ggerganov/llama.cpp
- FlashAttention — https://arxiv.org/abs/2205.14135
- FlashAttention-2 — https://arxiv.org/abs/2307.08691
- FlashAttention-3 — https://arxiv.org/abs/2407.08608

## Quantization

- GPTQ — https://arxiv.org/abs/2210.17323
- AWQ — https://arxiv.org/abs/2306.00978
- SmoothQuant — https://arxiv.org/abs/2211.10438
- QLoRA — https://arxiv.org/abs/2305.14314
- bitsandbytes — https://github.com/bitsandbytes-foundation/bitsandbytes

## RAG and evaluation

- Retrieval-Augmented Generation — https://arxiv.org/abs/2005.11401
- Ragas documentation — https://docs.ragas.io/
- BEIR benchmark — https://arxiv.org/abs/2104.08663
- LangSmith evaluation docs — https://docs.smith.langchain.com/evaluation

## Agents and workflows

- ReAct — https://arxiv.org/abs/2210.03629
- Toolformer — https://arxiv.org/abs/2302.04761
- OpenAI function calling guide — https://platform.openai.com/docs/guides/function-calling
- LangGraph documentation — https://docs.langchain.com/oss/python/langgraph/overview

---

# Final Mental Model

The roadmap compresses into this dependency chain:

```text
Foundation
→ understand one-token generation mechanically

Training
→ understand how base capability is created

Post-training
→ understand how behavior is shaped

Reasoning
→ understand inference-time computation and verification

Inference
→ understand latency, batching, memory, and throughput

Serving engines
→ choose the runtime based on workload constraints

KV cache / long context
→ control memory growth and context reliability

Quantization
→ reduce cost with measured quality tradeoffs

RAG
→ ground answers in external knowledge

Agents
→ build bounded tool-using workflows

Evaluation
→ measure quality, safety, latency, and cost

Production
→ deploy, monitor, secure, scale, and govern

Advanced tracks
→ multimodal, domain adaptation, security, research literacy, hardware-aware engineering
```

The professional standard is simple:

```text
No claims without eval.
No architecture without failure modes.
No optimization without measurement.
No agent without boundaries.
No RAG without retrieval metrics.
No serving decision without workload details.
No production system without observability.
```
