# LLM Systems Engineering Roadmap

> A professional roadmap for mastering large language model internals, training, post-training, inference, retrieval, agents, evaluation, and production architecture.

This repository is designed for engineers who want to move beyond surface-level LLM usage and build production-grade LLM systems with measurable quality, latency, cost, and reliability.

It is not a collection of model news.

It is not a prompt-engineering cookbook.

It is a systems roadmap.

The central idea:

```text
LLM competence = model internals + training logic + inference systems + retrieval architecture + agent control + evaluation discipline + production constraints
```

---

## Table of Contents

- [Who this roadmap is for](#who-this-roadmap-is-for)
- [What this roadmap covers](#what-this-roadmap-covers)
- [What this roadmap does not cover](#what-this-roadmap-does-not-cover)
- [Core philosophy](#core-philosophy)
- [Competency map](#competency-map)
- [Roadmap overview](#roadmap-overview)
- [Layer 1: LLM Foundations](#layer-1-llm-foundations)
- [Layer 2: Training Pipeline](#layer-2-training-pipeline)
- [Layer 3: Post-Training](#layer-3-post-training)
- [Layer 4: Reasoning Models](#layer-4-reasoning-models)
- [Layer 5: Inference Fundamentals](#layer-5-inference-fundamentals)
- [Layer 6: Serving Engines](#layer-6-serving-engines)
- [Layer 7: KV Cache and Long Context](#layer-7-kv-cache-and-long-context)
- [Layer 8: Quantization and Compression](#layer-8-quantization-and-compression)
- [Layer 9: RAG Systems](#layer-9-rag-systems)
- [Layer 10: Agentic Systems](#layer-10-agentic-systems)
- [Layer 11: Evaluation and Benchmarking](#layer-11-evaluation-and-benchmarking)
- [Layer 12: Production Architecture](#layer-12-production-architecture)
- [Advanced tracks](#advanced-tracks)
- [Master artifact portfolio](#master-artifact-portfolio)
- [Repository structure](#repository-structure)
- [Definition of done](#definition-of-done)
- [Recommended source map](#recommended-source-map)
- [Engineering checklists](#engineering-checklists)
- [How to use this roadmap](#how-to-use-this-roadmap)

---

## Who this roadmap is for

This roadmap is for:

- AI engineers
- ML engineers
- NLP engineers
- backend engineers moving into LLM systems
- technical leads responsible for GenAI architecture
- researchers who want stronger production intuition
- product-minded engineers building applied LLM systems
- infrastructure engineers working with GPU serving stacks

You should use this roadmap if your target is to build systems like:

- enterprise RAG platforms
- on-prem LLM deployments
- multi-model inference gateways
- AI agents with tools and approvals
- evaluation harnesses for LLM products
- private domain assistants
- document intelligence systems
- multimodal knowledge systems
- production LLM observability pipelines
- cost-controlled LLM serving infrastructure

---

## What this roadmap covers

This roadmap covers the full technical stack behind modern LLM systems:

```text
Text
→ Tokens
→ Transformer
→ Pretraining
→ Post-training
→ Reasoning
→ Inference runtime
→ Serving engine
→ KV cache
→ Quantization
→ Retrieval
→ Agents
→ Evaluation
→ Production architecture
```

It explains the mechanisms, not only the buzzwords.

Each layer includes:

- objective
- core concepts
- what to understand deeply
- implementation artifacts
- engineering decisions
- failure modes
- evaluation gates
- recommended resources

---

## What this roadmap does not cover

This roadmap does not focus on:

- daily model release news
- shallow prompt collections
- generic AI career advice
- vendor marketing claims
- no-code tool tutorials
- toy demos without evaluation
- “best model” lists without workload definition

The assumption is simple:

```text
A model is not good or bad in isolation.
A model is good or bad for a workload, under constraints, measured by an eval.
```

---

## Core philosophy

### 1. Learn mechanisms, not names

Do not memorize model names. Learn what changed.

```text
What changed in architecture?
What changed in data?
What changed in post-training?
What changed in inference?
What changed in memory layout?
What changed in evaluation?
```

Model names expire. Mechanisms compound.

---

### 2. Separate model science from system engineering

A capable LLM system is not just a strong model.

It is a controlled pipeline:

```text
model
+ tokenizer
+ chat template
+ retrieval
+ tools
+ serving engine
+ cache policy
+ eval set
+ observability
+ fallback logic
+ cost controls
+ safety boundaries
```

Most production failures happen outside the model weights.

---

### 3. Build artifacts, not opinions

For every layer, produce something measurable:

```text
benchmark
eval set
notebook
dashboard
architecture diagram
serving comparison
failure analysis
cost model
red-team suite
```

If your knowledge cannot produce an artifact, it is not operational yet.

---

### 4. Evaluate everything

Do not trust:

- one prompt
- one demo
- one leaderboard
- one benchmark
- one model card
- one latency number
- one anecdotal answer

Use evals, traces, failure categories, and regression tests.

---

### 5. Optimize for decision quality

The final goal is not to know more terms.

The final goal is to make better technical decisions:

```text
Should we fine-tune or use RAG?
Should we use vLLM or SGLang?
Should we quantize to INT4 or keep FP16?
Should we use long context or retrieval?
Should this be an agent or deterministic workflow?
Should this run on-prem or through an API?
Should we add a reranker?
Should we use a reasoning model?
```

---

## Competency map

### Level 0 — API user

Can call hosted APIs.

Typical abilities:

- writes prompts
- uses chat interfaces
- calls model endpoints
- adjusts temperature
- knows model names

Limit:

```text
Cannot explain or debug failures below the API layer.
```

---

### Level 1 — Prototype builder

Can build demos.

Typical abilities:

- builds simple RAG
- uses LangChain/LlamaIndex
- connects vector databases
- builds tool-calling examples
- creates chatbot demos

Limit:

```text
Often lacks evaluation, observability, failure analysis, and production constraints.
```

---

### Level 2 — LLM application engineer

Can build useful applications.

Typical abilities:

- designs retrieval pipelines
- builds structured prompts
- manages citations
- performs basic evals
- handles tool calling
- integrates with backend systems

Limit:

```text
May not deeply understand inference, KV cache, serving engines, or GPU cost.
```

---

### Level 3 — LLM systems engineer

Can build production systems.

Typical abilities:

- understands prefill/decode
- benchmarks inference
- chooses serving engines
- estimates KV cache memory
- evaluates quantization
- builds RAG evals
- instruments traces
- designs fallback paths
- controls latency and cost

Minimum serious professional level.

---

### Level 4 — LLM infrastructure engineer

Can optimize large-scale serving.

Typical abilities:

- operates vLLM/SGLang/TensorRT-LLM
- handles multi-GPU serving
- manages concurrency
- tunes batching
- handles prefix caching
- evaluates quantized kernels
- monitors GPU utilization
- designs model gateways
- handles autoscaling

---

### Level 5 — Research engineer

Can understand and modify methods.

Typical abilities:

- reads papers mechanically
- runs ablations
- modifies post-training recipes
- tests reasoning methods
- builds custom evals
- analyzes training/inference tradeoffs
- understands architecture deltas

---

### Level 6 — LLM architect

Can design organization-scale platforms.

Typical abilities:

- defines platform architecture
- governs model usage
- builds eval infrastructure
- designs multi-tenant systems
- manages security and compliance
- controls cost at scale
- aligns model strategy with business constraints

Target level.

---

## Roadmap overview

| Layer | Area | Core question |
|---|---|---|
| 1 | LLM Foundations | What happens during one token generation? |
| 2 | Training Pipeline | How are base models created? |
| 3 | Post-Training | How are models shaped into assistants? |
| 4 | Reasoning Models | How do models use extra compute to solve hard tasks? |
| 5 | Inference Fundamentals | Why is serving an LLM a systems problem? |
| 6 | Serving Engines | Which runtime fits which workload? |
| 7 | KV Cache and Long Context | What makes long context expensive and unreliable? |
| 8 | Quantization and Compression | How do we reduce cost without silent quality collapse? |
| 9 | RAG Systems | How do we ground outputs in external knowledge? |
| 10 | Agentic Systems | How do we safely connect models to tools and workflows? |
| 11 | Evaluation and Benchmarking | How do we measure quality, cost, latency, and safety? |
| 12 | Production Architecture | How do we deploy, monitor, scale, and govern LLM systems? |

---

# Layer 1: LLM Foundations

## Objective

Understand the core mechanics of a decoder-only LLM.

The minimum target:

```text
Given a prompt, explain exactly how the model turns text into token probabilities.
```

## Core concepts

### Tokenization

The model does not read words. It reads tokens.

A tokenizer converts text into integer IDs:

```text
"Large language models are useful"
→ [24513, 4221, 4981, 527, 5562]
```

Tokenization affects cost, context length, latency, multilingual quality, code handling, Arabic morphology, prompt compression, domain vocabulary, and retrieval chunk size.

A bad tokenizer can increase token count and damage quality, especially for morphologically rich languages.

Key rule:

```text
Never estimate LLM cost from word count.
Always measure with the model tokenizer.
```

### Embeddings

Token IDs are indices.

The model maps each token ID to a dense vector through an embedding table:

```text
vocabulary_size × hidden_dimension
```

The prompt becomes:

```text
sequence_length × hidden_dimension
```

### Transformer blocks

A decoder-only LLM is a stack of Transformer blocks:

```text
input
→ normalization
→ self-attention
→ residual connection
→ normalization
→ MLP
→ residual connection
→ output
```

Each block edits the representation. It does not rebuild everything from scratch.

### Self-attention

Self-attention lets each token route information from previous tokens.

Each token representation is projected into:

```text
Q = query
K = key
V = value
```

Intuition:

```text
Query = what this position is looking for
Key   = what each position offers for matching
Value = information carried if selected
```

Scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

### Causal masking

Decoder-only models cannot look into the future.

For tokens:

```text
A B C D
```

position `C` can attend to:

```text
A B C
```

but not:

```text
D
```

This enables next-token training without information leakage.

### Multi-head attention

Multiple attention heads let the model route different information patterns in parallel: syntax, long-range reference, formatting, code indentation, list structure, and mathematical dependencies.

### MQA and GQA

Classic multi-head attention stores separate keys and values for every attention head. That is expensive during inference.

Modern models often use:

- MQA: many query heads share one key/value head
- GQA: groups of query heads share key/value heads

Why this matters:

```text
Fewer KV heads → smaller KV cache → better serving scalability
```

### MLP blocks

Attention mixes information across token positions.

MLP layers transform each token representation independently.

```text
Attention = token-to-token communication
MLP       = token-wise feature transformation
```

Many LLM parameters live in MLP blocks.

### Positional encoding

Attention alone does not know order.

Modern LLMs commonly use RoPE. RoPE injects position by rotating query and key vectors in a position-dependent way.

Important implication:

```text
Long-context behavior is partly constrained by positional encoding design.
```

### Logits and decoding

The final hidden state is projected into vocabulary logits:

```text
hidden_dim → vocab_size
```

Softmax turns logits into probabilities. Decoding chooses the next token.

Common decoding methods:

- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling
- repetition penalties
- constrained decoding

### KV cache

During generation, the model stores previously computed key/value tensors. This avoids recomputing all previous tokens.

KV cache grows with:

```text
batch_size × context_length × layers × KV_heads × head_dim × bytes_per_value
```

This is one of the most important memory bottlenecks in LLM serving.

## What to implement

Build a tiny decoder-only model.

Minimum components:

```text
tokenizer
embedding layer
causal self-attention
MLP
residual connections
normalization
logits head
sampling loop
```

## Practical exercises

### Exercise 1: Tokenizer comparison

Compare token counts across English, Arabic, mixed Arabic-English, Python code, JSON, and legal text.

Record:

```text
characters
words
tokens
tokens per word
strange splits
```

### Exercise 2: Decode behavior

Run one model with:

```text
temperature = 0
temperature = 0.3
temperature = 0.8
top_p = 0.9
top_k = 50
```

Observe correctness, variation, repetition, hallucination, and formatting stability.

### Exercise 3: KV cache intuition

Measure memory and latency at:

```text
1k context
4k context
16k context
32k context
```

Track time to first token, time per output token, GPU memory, and throughput.

## Evaluation gate

You pass this layer if you can explain:

```text
text → tokens → embeddings → Transformer blocks → logits → probabilities → next token
```

without hand-waving.

---

# Layer 2: Training Pipeline

## Objective

Understand how base LLM capability is created before instruction tuning.

A base model is not yet a helpful assistant. It is a statistical language model trained to predict the next token.

## Core pipeline

```text
raw data
→ filtering
→ deduplication
→ classification
→ mixture design
→ tokenizer training
→ sequence packing
→ pretraining
→ checkpointing
→ validation
→ contamination checks
→ base model release/evaluation
```

## Data construction

Training data quality dominates model behavior.

Sources may include web pages, books, code, academic text, documentation, forums, math data, multilingual corpora, synthetic data, and domain-specific corpora.

Data quality issues include spam, boilerplate, duplicated pages, machine-generated junk, toxic content, benchmark contamination, stale facts, low-quality translations, formatting noise, and personally identifiable information.

Key principle:

```text
Pretraining data is not just fuel.
It is the model's compressed world.
```

## Deduplication

Deduplication reduces repeated content.

Why it matters:

- prevents memorization
- improves data diversity
- reduces overfitting
- reduces benchmark leakage
- improves compute efficiency

Types:

- exact deduplication
- near-duplicate detection
- document-level deduplication
- paragraph-level deduplication
- code clone detection

## Data mixture

Not all data should have equal weight.

A data mixture controls how much of each domain the model sees.

Examples:

```text
web text
code
math
books
scientific papers
multilingual content
instruction-like content
```

Data mixture affects code ability, reasoning, multilingual quality, factual recall, style, toxicity, and domain competence.

## Tokenizer training

Tokenizer decisions affect the whole model.

Consider vocabulary size, BPE vs Unigram vs WordPiece, byte fallback, multilingual coverage, code tokens, special tokens, whitespace behavior, Arabic and dialect handling.

A tokenizer is hard to change after training.

Changing tokenizer usually means training or adapting the model again.

## Training objective

Most decoder-only LLMs use next-token prediction.

The model minimizes cross-entropy loss:

```text
good prediction → low loss
bad prediction  → high loss
```

Loss is useful but incomplete.

A lower loss does not automatically mean better instruction following, better reasoning, better safety, better RAG behavior, or better tool use.

## Scaling laws

Scaling laws relate model size, dataset size, compute budget, and loss.

They help answer:

```text
Given fixed compute, should we train a larger model on fewer tokens or a smaller model on more tokens?
```

Important principle:

```text
Compute-optimal training is a resource allocation problem.
```

## Optimization

Important components:

- AdamW
- learning rate schedule
- warmup
- gradient clipping
- weight decay
- mixed precision
- gradient accumulation
- batch size
- checkpointing

Training instability can come from bad data, bad learning rate, optimizer settings, numerical overflow, distributed training bugs, tokenizer/data mismatch, or corrupted batches.

## Distributed training

Large models require distributed training.

Common strategies:

- data parallelism
- tensor parallelism
- pipeline parallelism
- sequence parallelism
- ZeRO-style optimizer sharding
- activation checkpointing

Training is constrained by GPU memory, interconnect bandwidth, compute utilization, checkpoint I/O, failure recovery, and cluster scheduling.

## What to implement

Build a mini pretraining pipeline:

```text
collect text
clean text
train tokenizer
pack sequences
train tiny model
track loss
sample generations
evaluate basic capability
```

## Evaluation gate

You pass this layer if you can read a model technical report and identify:

```text
data recipe
token budget
model size
architecture
training objective
compute estimate
evaluation setup
contamination risks
```

---

# Layer 3: Post-Training

## Objective

Understand how base models become useful assistants.

Base models complete text. Post-trained models follow instructions.

## Post-training pipeline

```text
base model
→ supervised fine-tuning
→ preference optimization
→ reinforcement learning / direct optimization
→ safety tuning
→ refusal calibration
→ formatting alignment
→ evaluation
```

## Supervised fine-tuning

SFT trains the model on instruction-response pairs.

It teaches instruction following, chat behavior, formatting, domain response style, role behavior, and basic helpfulness.

But SFT alone can teach imitation, not necessarily preference quality.

## Preference optimization

Preference data contains comparisons:

```text
prompt
chosen answer
rejected answer
```

The model learns which answer is preferred.

Methods include:

- RLHF
- DPO
- IPO
- KTO
- ORPO
- RLAIF

## RLHF

RLHF often uses:

```text
preference data
→ reward model
→ policy optimization
```

Benefits:

- improves helpfulness
- aligns with human preference
- improves conversational behavior

Risks:

- reward hacking
- over-optimization
- verbosity bias
- style over truth
- calibration damage

## DPO

Direct Preference Optimization removes the separate reward-model training loop.

It directly optimizes the model using preference pairs.

Benefits:

- simpler than RLHF
- easier to implement
- widely used for alignment experiments

Limit:

```text
Quality depends heavily on preference data quality.
```

## RLVR and GRPO

RL with verifiable rewards is important for reasoning tasks.

A reward is verifiable when correctness can be checked automatically.

Examples:

- math answer correctness
- code tests passing
- exact symbolic results
- game outcomes
- tool-verified facts

This is more reliable than subjective reward for many reasoning tasks.

## Safety tuning

Safety tuning shapes refusal behavior, policy adherence, harmful request handling, uncertainty expression, tool permission behavior, and sensitive data handling.

Bad safety tuning can cause over-refusal, under-refusal, evasive answers, false confidence, and degraded utility.

## What to implement

Run a small post-training experiment:

```text
base model
→ SFT
→ preference optimization
→ before/after eval
```

Track instruction following, factual accuracy, formatting, refusal behavior, hallucination, verbosity, and domain performance.

## Evaluation gate

You pass this layer if you can decide between:

```text
prompting
RAG
SFT
LoRA
DPO
continued pretraining
```

based on the actual failure mode.

---

# Layer 4: Reasoning Models

## Objective

Understand models that spend additional inference compute to solve harder tasks.

## Core idea

A reasoning model is not just a model that outputs long explanations.

Reasoning systems often involve longer internal deliberation, verifiable rewards, search, self-consistency, verifier models, test-time compute, and specialized post-training.

## Chain-of-thought

Chain-of-thought encourages intermediate reasoning.

It can help with math, logic, planning, code, and multi-step questions.

But it can fail when reasoning is ungrounded, the model fabricates steps, the task requires external knowledge, the chain is persuasive but wrong, or hidden assumptions go unchecked.

## Test-time compute

Test-time compute means spending more inference resources for better answers.

Examples:

- generate multiple candidates
- vote across answers
- use a verifier
- search over reasoning paths
- run code/tools
- critique and revise

Tradeoff:

```text
better quality potential
vs
higher latency and cost
```

## Verifiers

A verifier scores or checks candidate answers.

Types:

- outcome verifier
- process verifier
- unit test verifier
- symbolic verifier
- retrieval-grounded verifier
- human verifier

Verifiers work best when correctness is measurable.

## Overthinking failure

Reasoning models can overthink.

Symptoms:

- unnecessary long reasoning
- changing correct answers
- unstable final answer
- higher cost without better quality
- worse performance on simple tasks

Decision rule:

```text
Use reasoning models where extra compute changes accuracy.
Do not use them by default.
```

## What to implement

Create a reasoning eval harness:

```text
question
baseline answer
reasoning answer
tool-verified answer
latency
cost
correctness
failure mode
```

## Evaluation gate

You pass this layer if you can classify a task into:

```text
direct answer
retrieval required
tool required
reasoning required
human approval required
```

---

# Layer 5: Inference Fundamentals

## Objective

Understand LLM inference as a systems problem.

## Request lifecycle

```text
request arrives
→ tokenize
→ build prompt/chat template
→ prefill
→ first token
→ decode loop
→ stream output
→ stop condition
→ log trace
```

## Prefill

Prefill processes the input prompt.

It creates KV cache for prompt tokens.

Main metric:

```text
TTFT = time to first token
```

Long prompts increase TTFT.

## Decode

Decode generates output one token at a time.

Main metric:

```text
TPOT = time per output token
```

Decode is often constrained by memory bandwidth and KV cache reads.

## Throughput vs latency

Throughput:

```text
tokens per second
```

Latency:

```text
how long one user waits
```

They are not the same.

A system can have high throughput and poor user latency.

Measure both.

## Batching

Batching improves GPU utilization.

But LLM requests have variable lengths.

Static batching wastes capacity.

Continuous batching dynamically adds and removes requests.

This improves utilization under live traffic.

## Metrics

Track:

```text
TTFT
TPOT
end-to-end latency
tokens/sec
requests/sec
p50 latency
p95 latency
p99 latency
GPU utilization
VRAM usage
queue time
error rate
```

## What to implement

Build an inference benchmark suite.

Test:

```text
single request
many concurrent requests
short prompt
long prompt
short output
long output
streaming
non-streaming
```

## Evaluation gate

You pass this layer if you never report “tokens/sec” without workload definition.

---

# Layer 6: Serving Engines

## Objective

Choose the correct runtime for a workload.

## Engine categories

### Local developer engines

Examples:

- Ollama
- llama.cpp

Best for local experiments, CPU/Mac workflows, edge deployments, and quick testing.

Not ideal for high-concurrency production serving, advanced GPU scheduling, or multi-tenant inference platforms.

### Production open-source serving engines

Examples:

- vLLM
- SGLang
- Hugging Face TGI
- LMDeploy

Best for high-throughput serving, OpenAI-compatible APIs, batching, prefix caching, multi-GPU serving, and production model endpoints.

### Vendor-optimized engines

Example:

- TensorRT-LLM

Best for NVIDIA GPU optimization, maximum performance, controlled deployment environments, and latency-sensitive workloads.

Tradeoff:

```text
higher complexity
more hardware-specific optimization
```

## Selection criteria

Choose serving engine based on:

```text
model architecture support
hardware
quantization format
latency target
throughput target
context length
concurrency
structured output needs
LoRA serving
multi-GPU support
observability
operational complexity
team skill
```

## Serving comparison matrix

| Engine | Best use | Strength | Risk |
|---|---|---|---|
| vLLM | General production serving | Throughput, ecosystem | Model-specific edge cases |
| SGLang | Structured/high-performance workloads | Prefix reuse, structured generation | Operational learning curve |
| TensorRT-LLM | NVIDIA-optimized serving | Performance | Complexity |
| llama.cpp | Local/edge | Portability | Not ideal for high-concurrency serving |
| Ollama | Developer UX | Simplicity | Limited production control |

## Evaluation gate

You pass this layer if you can justify engine choice using constraints, not preference.

---

# Layer 7: KV Cache and Long Context

## Objective

Understand the real cost of context length.

## KV cache memory

KV cache stores previous keys and values for every generated/request token.

Memory grows with:

```text
batch_size
context_length
num_layers
num_kv_heads
head_dim
dtype_bytes
```

Simplified:

```text
KV memory ≈ 2 × batch × seq_len × layers × kv_heads × head_dim × bytes
```

The factor `2` is for K and V.

## Why long context is hard

Long context causes:

- higher prefill cost
- larger KV cache
- higher memory pressure
- slower scheduling
- lost-in-the-middle behavior
- attention dilution
- more prompt injection surface
- more irrelevant information
- higher cost

## Prefix caching

Prefix caching reuses KV cache for shared prompt prefixes.

Useful for repeated system prompts, few-shot examples, static policy blocks, agent frameworks, repeated document prefixes, and multi-turn sessions.

## Context is not memory

A 128k context window means the model can accept 128k tokens.

It does not mean it can reliably reason over all 128k tokens.

Quality still depends on position sensitivity, retrieval quality, prompt structure, instruction hierarchy, distractor density, and model training.

## RAG vs long context

Use long context when:

- all context is relevant
- order matters
- context changes per request
- retrieval misses critical details

Use RAG when:

- corpus is large
- only small slices are relevant
- freshness matters
- citations matter
- permission control matters
- cost matters

## What to implement

Build a KV cache calculator.

Inputs:

```text
layers
kv_heads
head_dim
dtype
batch size
context length
```

Output:

```text
estimated KV memory
max concurrency
memory risk
```

## Evaluation gate

You pass this layer if you can estimate memory before deployment.

---

# Layer 8: Quantization and Compression

## Objective

Reduce memory and cost without destroying quality.

## Numeric formats

Common formats:

- FP32
- FP16
- BF16
- FP8
- INT8
- INT4

Lower precision reduces memory.

But it introduces numerical error.

Quantization is controlled damage.

## What can be quantized

### Weights

Most common. Reduces model memory.

### Activations

More complex. Can improve throughput if kernels support it.

### KV cache

Reduces memory for long context and high concurrency. Can damage long-context quality.

## Common methods

### GPTQ

Post-training weight quantization. Often used for GPU inference.

### AWQ

Activation-aware weight quantization. Often strong for preserving quality in low-bit inference.

### GGUF

Common format for llama.cpp ecosystem. Useful for local and edge deployment.

### SmoothQuant

Balances activation and weight quantization difficulty.

### QLoRA

Uses quantized base weights for memory-efficient fine-tuning.

## Benchmark dimensions

Measure:

```text
quality
latency
throughput
VRAM
TTFT
TPOT
format stability
code correctness
reasoning accuracy
RAG faithfulness
tool-call validity
```

Do not evaluate quantization only with perplexity.

## What to implement

Run:

```text
FP16 baseline
INT8
INT4 GPTQ
INT4 AWQ
GGUF
KV INT8
```

Compare against domain evals.

## Evaluation gate

You pass this layer if you can say exactly what was quantized, how, and what quality changed.

---

# Layer 9: RAG Systems

## Objective

Build retrieval systems that ground LLM outputs in external knowledge.

## Basic RAG pipeline

```text
documents
→ parsing
→ cleaning
→ chunking
→ embedding
→ indexing
→ retrieval
→ reranking
→ prompt construction
→ generation
→ citation validation
→ evaluation
```

## Chunking

Chunking controls what the retriever can find.

Bad chunking causes missing context, fragmented answers, irrelevant retrieval, citation mismatch, and hallucination.

Chunking strategies:

- fixed-size chunks
- semantic chunks
- section-based chunks
- parent-child chunks
- sliding windows
- page-level chunks

## Retrieval methods

### BM25

Good for exact terms, IDs, names, legal references, rare words.

### Dense retrieval

Good for semantic similarity.

### Hybrid retrieval

Combines lexical and semantic retrieval.

Often stronger than either alone.

### RRF

Reciprocal Rank Fusion combines ranked lists from multiple retrievers.

Simple and effective.

## Reranking

A reranker scores query-document relevance more precisely.

Typical flow:

```text
retrieve top 50
→ rerank
→ keep top 5-10
```

Reranking improves precision at the cost of latency.

## RAG failure modes

Failures can happen at:

```text
parsing
chunking
embedding
indexing
retrieval
reranking
prompt construction
generation
citation validation
```

Debug the stage.

Do not blame the model first.

## What to implement

Build a RAG system with:

```text
BM25
dense retrieval
hybrid retrieval
RRF
reranker
citations
eval set
trace logging
```

## Evaluation gate

You pass this layer if you can separate retrieval failure from generation failure.

---

# Layer 10: Agentic Systems

## Objective

Build controlled tool-using LLM systems.

## Agent definition

An agent is a system where an LLM can choose actions.

Examples:

- call tools
- query databases
- browse documents
- write files
- send emails
- schedule actions
- run code
- ask for approval

The danger:

```text
More autonomy means more failure surface.
```

## Workflows vs agents

Use deterministic workflows when the steps are known.

Use agents when the path must be chosen dynamically.

Rule:

```text
Workflow first.
Agent only where decision flexibility is needed.
```

## Core patterns

### Router

Chooses which path or model to use.

### Tool caller

Calls external functions with structured arguments.

### Planner

Breaks a task into steps.

### Executor

Performs actions.

### Verifier

Checks output.

### Human approval gate

Stops risky actions before execution.

## State and memory

Agents need state.

State may include user goal, current plan, completed steps, tool outputs, constraints, errors, budget, and approval status.

Memory must be controlled.

Unbounded memory creates confusion and security risk.

## Failure modes

- infinite loops
- tool misuse
- wrong tool arguments
- stale memory
- prompt injection
- unauthorized action
- hidden cost explosion
- hallucinated tool results
- invalid final answer

## What to implement

Build a bounded agent:

```text
planner
tool registry
schema validation
executor
verifier
retry limit
cost limit
approval gate
trace log
```

## Evaluation gate

You pass this layer if your agent can fail safely.

---

# Layer 11: Evaluation and Benchmarking

## Objective

Measure LLM system quality before users discover failures.

## Evaluation types

### Model evals

Measure model behavior directly.

Examples: factuality, reasoning, coding, summarization, instruction following.

### RAG evals

Measure retrieval and grounded generation.

Metrics:

- context precision
- context recall
- faithfulness
- answer relevance
- citation correctness

### Agent evals

Measure action quality.

Metrics:

- task success
- tool correctness
- invalid tool calls
- loop rate
- approval violations
- cost per task

### Production evals

Measure real-world operation.

Metrics:

- latency
- error rate
- user correction rate
- fallback rate
- escalation rate
- cost
- safety incidents

## Golden datasets

A golden dataset is a curated set of cases representing expected behavior.

It should include easy cases, hard cases, edge cases, adversarial cases, outdated info cases, ambiguous cases, negative cases, and refusal cases.

## LLM-as-judge

LLM judges can help, but they must be controlled.

Use clear rubrics, pairwise comparisons, calibration examples, human-reviewed samples, and judge agreement checks.

Never blindly trust judge scores.

## Regression testing

Every production change should run evals.

Changes include model update, prompt change, retrieval change, reranker change, chunking change, tool change, quantization change, and serving engine change.

## What to implement

Build an eval harness:

```text
dataset
input
expected behavior
retrieved context
model output
judge rubric
latency
cost
failure category
release decision
```

## Evaluation gate

You pass this layer if every major system change has a measurable before/after result.

---

# Layer 12: Production Architecture

## Objective

Design LLM systems that survive real users, real latency, real cost, and real failure.

## Reference architecture

```text
client
→ API gateway
→ authentication
→ rate limiting
→ request logger
→ prompt builder
→ router
→ retrieval service
→ tool service
→ model gateway
→ serving engine
→ response validator
→ trace store
→ eval pipeline
→ monitoring dashboard
```

## Model gateway

A model gateway abstracts access to multiple models.

It handles routing, fallbacks, retries, budget policies, provider abstraction, model versioning, logging, and safety checks.

## Observability

Track:

```text
prompt
model
version
latency
tokens in
tokens out
retrieved chunks
tool calls
errors
cost
user feedback
eval score
```

Without traces, debugging becomes guessing.

## Security

Production LLM systems must handle prompt injection, indirect prompt injection, tool abuse, data exfiltration, PII leakage, retrieval poisoning, unauthorized access, tenant isolation, and audit logging.

## Cost control

Cost comes from input tokens, output tokens, model size, inference engine, GPU utilization, concurrency, reranking, embeddings, tool calls, retries, logging, and evaluation.

Cost must be measured per task, not only per token.

## What to implement

Design a complete production architecture document:

```text
system diagram
data flow
model flow
failure modes
security controls
observability plan
cost model
eval gates
scaling plan
rollback plan
```

## Evaluation gate

You pass this layer if you can review an LLM architecture and identify reliability, security, cost, and quality risks.

---

# Advanced tracks

## Advanced Track A: Multimodal LLMs

Learn:

- vision-language models
- audio-language models
- document understanding
- OCR pipelines
- image embeddings
- video frame sampling
- multimodal RAG
- visual grounding
- multimodal evals

Build:

```text
PDF/image ingestion
OCR
layout extraction
visual chunking
text + image retrieval
grounded answer generation
citation to page/region
```

## Advanced Track B: Domain Adaptation

Learn:

- prompt adaptation
- RAG
- SFT
- LoRA
- QLoRA
- continued pretraining
- domain-specific tokenization
- ontology grounding
- terminology normalization
- legal/medical/financial evals

Decision hierarchy:

```text
prompting
→ RAG
→ SFT/LoRA
→ continued pretraining
```

Continued pretraining is expensive and should not be the default.

## Advanced Track C: LLM Security

Learn:

- prompt injection
- jailbreaks
- indirect prompt injection
- retrieval poisoning
- tool abuse
- sandboxing
- output validation
- permission boundaries
- audit trails
- secure agent design

Build:

```text
red-team suite
prompt injection tests
tool misuse tests
retrieval poisoning tests
PII leakage tests
policy bypass tests
```

## Advanced Track D: Hardware-Aware LLM Engineering

Learn:

- HBM bandwidth
- tensor cores
- CUDA kernels
- FlashAttention
- NCCL
- NVLink
- PCIe
- tensor parallelism
- pipeline parallelism
- expert parallelism
- GPU memory fragmentation

Build:

```text
hardware fit calculator
model memory estimator
KV cache estimator
throughput benchmark
GPU utilization dashboard
```

## Advanced Track E: Research Literacy

Use this template for every paper:

```text
Claim:
Mechanism:
What changed:
What stayed constant:
Dataset:
Compute:
Ablation:
Metric:
Weakness:
Reproducibility:
Production implication:
```

The goal is to identify mechanism, not memorize title.

---

# Master artifact portfolio

Build these artifacts to prove competence.

| ID | Artifact | Purpose |
|---|---|---|
| 01 | Tiny Transformer | Understand token generation mechanically |
| 02 | Tokenizer Comparison Notebook | Measure tokenizer impact across languages/domains |
| 03 | Mini Pretraining Pipeline | Understand data, tokenization, loss, and sampling |
| 04 | SFT Experiment | Learn instruction tuning |
| 05 | DPO/Preference Experiment | Learn preference optimization |
| 06 | Reasoning Eval Harness | Compare normal vs reasoning models |
| 07 | Inference Benchmark Suite | Measure TTFT, TPOT, latency, throughput |
| 08 | Serving Engine Matrix | Compare vLLM, SGLang, TensorRT-LLM, llama.cpp |
| 09 | KV Cache Calculator | Estimate serving memory |
| 10 | Quantization Benchmark | Measure quality/cost tradeoffs |
| 11 | Production RAG System | Ground answers with retrieval and citations |
| 12 | Agent Workflow | Build controlled tool use |
| 13 | Eval Dashboard | Track quality, latency, cost, safety |
| 14 | Production Architecture Diagram | Design deployable platform |
| 15 | Security Red-Team Suite | Test prompt injection and tool abuse |
| 16 | Cost Model | Estimate per-task and platform-level cost |
| 17 | Paper Review Database | Build research literacy |

---

# Repository structure

Recommended structure:

```text
llm-systems-engineering-roadmap/
│
├── README.md
├── LICENSE
├── roadmap/
│   ├── 01_llm_foundations.md
│   ├── 02_training_pipeline.md
│   ├── 03_post_training.md
│   ├── 04_reasoning_models.md
│   ├── 05_inference_fundamentals.md
│   ├── 06_serving_engines.md
│   ├── 07_kv_cache_long_context.md
│   ├── 08_quantization_compression.md
│   ├── 09_rag_systems.md
│   ├── 10_agentic_systems.md
│   ├── 11_evaluation_benchmarking.md
│   └── 12_production_architecture.md
│
├── artifacts/
│   ├── tiny_transformer/
│   ├── tokenizer_comparison/
│   ├── mini_pretraining/
│   ├── post_training/
│   ├── reasoning_eval/
│   ├── inference_benchmark/
│   ├── kv_cache_calculator/
│   ├── quantization_benchmark/
│   ├── rag_system/
│   ├── agent_workflow/
│   ├── eval_dashboard/
│   └── production_architecture/
│
├── templates/
│   ├── paper_review_template.md
│   ├── model_eval_template.md
│   ├── rag_eval_template.md
│   ├── agent_eval_template.md
│   ├── architecture_review_template.md
│   └── cost_model_template.md
│
├── resources/
│   ├── papers.md
│   ├── docs.md
│   ├── courses.md
│   ├── tools.md
│   └── benchmarks.md
│
└── checklists/
    ├── model_selection_checklist.md
    ├── rag_production_checklist.md
    ├── agent_safety_checklist.md
    ├── inference_benchmark_checklist.md
    ├── quantization_checklist.md
    └── production_readiness_checklist.md
```

---

# Definition of done

You are not done when you read the chapters.

You are done when you can produce these outputs.

## Foundation done

```text
Can implement and explain a tiny Transformer.
Can trace one token generation.
Can explain tokenization, logits, sampling, and KV cache.
```

## Training done

```text
Can build a mini pretraining loop.
Can explain data mixture, loss, scaling, and contamination.
```

## Post-training done

```text
Can compare SFT, RLHF, DPO, GRPO, and RLAIF.
Can choose adaptation method based on failure mode.
```

## Reasoning done

```text
Can evaluate when reasoning models help.
Can measure accuracy vs latency/cost.
```

## Inference done

```text
Can benchmark TTFT, TPOT, throughput, p95 latency, and VRAM.
```

## Serving done

```text
Can choose a serving engine based on workload and hardware.
```

## KV cache done

```text
Can estimate KV cache memory and explain long-context tradeoffs.
```

## Quantization done

```text
Can evaluate quantization quality against domain tasks.
```

## RAG done

```text
Can build and debug hybrid retrieval with citations and evals.
```

## Agents done

```text
Can build bounded tool-using workflows with safe failure behavior.
```

## Evaluation done

```text
Can build regression evals and release gates.
```

## Production done

```text
Can design secure, observable, scalable LLM architecture.
```

---

# Recommended source map

## Foundations

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Hugging Face Tokenizer Summary: https://huggingface.co/docs/transformers/tokenizer_summary
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- PyTorch Transformer Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html

## Training and post-training

- Hugging Face TRL: https://huggingface.co/docs/trl/index
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- DPO: https://arxiv.org/abs/2305.18290
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

## Inference and serving

- vLLM Docs: https://docs.vllm.ai/en/latest/
- SGLang Docs: https://docs.sglang.ai/
- TensorRT-LLM Docs: https://docs.nvidia.com/tensorrt-llm/index.html
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Hugging Face TGI: https://huggingface.co/docs/text-generation-inference/index

## RAG and evaluation

- Ragas Metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- BEIR Benchmark: https://github.com/beir-cellar/beir
- MS MARCO: https://microsoft.github.io/msmarco/
- Sentence Transformers: https://www.sbert.net/
- ClawBench (real-world browser-agent evaluation): [paper](https://arxiv.org/abs/2604.08523) · [project](https://claw-bench.com/) · [code](https://github.com/TIGER-AI-Lab/ClawBench)

## Agents

- OpenAI Function Calling / Tools: https://platform.openai.com/docs/guides/function-calling
- LangGraph Workflows and Agents: https://docs.langchain.com/oss/python/langgraph/workflows-agents
- LangGraph Memory: https://docs.langchain.com/oss/python/langgraph/memory

## Security

- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework

---

# Engineering checklists

## Model selection checklist

```text
task type
language/domain
context length
latency target
cost target
quality target
tool use needed
reasoning needed
deployment mode
data privacy constraints
fine-tuning need
serving engine compatibility
quantization support
eval result
```

## RAG production checklist

```text
document parser tested
chunking strategy validated
metadata schema defined
hybrid retrieval implemented
reranker tested
citations validated
permission filters enforced
freshness handled
RAG eval set built
retrieval failures categorized
generation failures categorized
latency measured
cost measured
```

## Agent safety checklist

```text
tools have schemas
arguments validated
permissions enforced
dangerous actions require approval
retry limits exist
budget limits exist
tool outputs are logged
state is inspectable
prompt injection tests exist
fallback path exists
human escalation exists
```

## Inference benchmark checklist

```text
model version
precision
serving engine
GPU type
batch size
concurrency
prompt length
output length
TTFT
TPOT
p50 latency
p95 latency
p99 latency
tokens/sec
VRAM usage
GPU utilization
error rate
```

## Quantization checklist

```text
baseline measured
method identified
weights/activations/KV specified
calibration data documented
serving engine compatible
quality evaluated
latency evaluated
VRAM evaluated
hard cases tested
format stability tested
rollback available
```

## Production readiness checklist

```text
authentication
authorization
tenant isolation
rate limiting
prompt logging policy
PII policy
retrieval permissions
model fallback
eval gate
monitoring
alerts
cost dashboard
security tests
rollback plan
incident response
```

---

# How to use this roadmap

Do not read it passively.

Use this loop:

```text
study one layer
→ implement one artifact
→ measure it
→ write failure notes
→ create decision rules
→ move to next layer
```

For every topic, produce:

```text
1. mechanism explanation
2. code or architecture artifact
3. benchmark or eval
4. failure mode list
5. decision rule
```

The roadmap is complete only when it changes your engineering decisions.

---

# Final compression

```text
LLM foundations teach how tokens become predictions.
Training teaches where base capability comes from.
Post-training teaches how behavior is shaped.
Reasoning teaches when extra inference compute helps.
Inference teaches why latency and memory dominate.
Serving engines teach how runtime choices affect production.
KV cache teaches why context is expensive.
Quantization teaches how to trade precision for cost.
RAG teaches how to ground outputs.
Agents teach how to connect models to actions.
Evaluation teaches how to know if anything works.
Production architecture teaches how to make it survive real usage.
```

The professional standard is not “I know LLMs.”

The professional standard is:

```text
I can design, measure, debug, and operate LLM systems under real constraints.
```
