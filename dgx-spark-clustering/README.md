# Project DGX Clustering Proof of Concept

Date: Nov 30, 2025

***

## Resources

*   Lama3.1 details: [Llama 3.1 - 405B, The text refers to **Llama 3.1 405B**, the flagship model in Meta's family of large language models (LLMs).](https://huggingface.co/blog/llama31)
*   [70B & 8B with multilinguality and long context](https://huggingface.co/blog/llama31)
*   Int4 Quantized version is here: <https://huggingface.co/RedHatAI/Meta-Llama-3.1-405B-Instruct-quantized.w4a16>
*   Neural Magic was [bought](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud) by RedHat and is now **redHatAI** in Jan 2025
*   **Nvidia deep learning containers resources: <https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html>**

---

## Section 1: Executive Summary

The Goal: We are connecting two NVIDIA DGX Spark units (Grace Blackwell architecture) using a direct 200GbE QSFP56 interconnect to create a unified "Super Node." This cluster effectively pools the resources of both machines, unlocking capabilities that are impossible on a single device.

The Challenge: A single DGX Spark has 128GB of Unified Memory. While powerful, this is insufficient to load frontier-class models like Llama 3.1 405B, which requires ~**203GB** even when heavily quantized.

The Solution: By clustering two units via high-speed NVLink/NCCL protocols, we aggregate the memory to 256GB and double the compute power (2 PFLOPS). This Proof of Concept (PoC) will demonstrate:

1.  **Memory Pooling:** Running a 405B parameter model that physically cannot fit on one machine.
2.  **Tensor Parallelism:** Splitting the computational load of image generation (Flux.1) across two GPUs to halve inference latency.
3.  **High-Speed Interconnect:** Validating the 200Gbps (24GB/s) bandwidth between desktop units.

## Section 2: Prep Instructions (The "Homework")

*Hi Cherif, before we meet to cluster our DGX Spark, please run these steps on your machine. This will get the heavy downloads (260GB+) and compilations out of the way so we can start clustering immediately.*

**Total Est. Time:** ~1.5 Hours (mostly waiting for downloads).

### Part 1: The Big Downloads (Run these in the background)

*These take the longest. Start them first.*

#### 1. Download the Llama 3.1 405B Model (230 GB)

We need the specific 4-bit quantized version to fit the model across our two machines.

```bash
# Install the downloader tool
pip install -U "huggingface_hub[cli]"

# Create directory
mkdir -p ~/models

# Start the download (This will take a while)
hf download neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16 \
--local-dir ~/models/Meta-Llama-3.1-405B-Instruct-INT4 --exclude "*.pth"

````


#### 2\. Pull the Docker Images (32 GB)

Add your user to the docker group if not done

``` bash
> sudo usermod -aG docker $USER

```

Login to new bash to activate group membership

``` bash
> bash  
> groups  // checks current group, verify no docker
jazracherif adm sudo audio dip plugdev users lpadmin  
> newgrp docker  
> groups // shows docker group
docker adm sudo audio dip plugdev users lpadmin jazracherif

```

We need the Blackwell-optimized containers.

``` bash
# Inference Engine
docker pull nvcr.io/nvidia/vllm:25.09-py3

# Dev Environment (Must be 25.10 for CUDA 13 support)
docker pull nvcr.io/nvidia/pytorch:25.10-py3

```

### Part 2: Python Setup

Installs the tools for the Flux Image Generation demo.

Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install from requirement_full for the versioned package
``` bash
pip install -r requirements_full.txt
```

### Part 3: Compiling NCCL (The Communication Layer)

NVIDIA DGX tutorial: <https://build.nvidia.com/spark/nccl/overview>

Nccl repo: <https://github.com/NVIDIA/nccl?tab=readme-ov-file>

We need to build the high-speed interconnect drivers from scratch to match your OS kernel.

#### 1\. Install Build Tools

``` bash
sudo apt update
sudo apt install -y build-essential devscripts debhelper fakeroot git libopenmpi-dev openmpi-bin

```

#### 2\. Build NCCL

``` bash
mkdir -p ~/SourceCodeCompiled
cd ~/SourceCodeCompiled
git clone [https://github.com/NVIDIA/nccl.git](https://github.com/NVIDIA/nccl.git)
cd nccl
make -j src.build

```

#### 3\. Configure Environment Variables (Permanent)

Run this to add the paths to your profile so the system always finds them.

``` bash
cat << 'EOF' >> ~/.bashrc

# --- DGX Cluster Config ---
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/SourceCodeCompiled/nccl/build" 
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib:$LD_LIBRARY_PATH"
# --------------------------
EOF

# Apply changes now
source ~/.bashrc

```

### Part 4: Compiling the Speedometer (NCCL Tests)

<https://github.com/nvidia/nccl-tests>

This tool allows us to verify the 200GbE connection speed once I plug in.

``` bash
cd ~/SourceCodeCompiled
git clone [https://github.com/NVIDIA/nccl-tests.git](https://github.com/NVIDIA/nccl-tests.git)
cd nccl-tests

# Compile
make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME

```

Verify it works:

Run this quick command. It should output a table of results (even if they are low numbers for now) without crashing.

``` bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

```

## Section 3: "Game Day" Execution Guide

*Once I arrive with my DGX, here is the roadmap of what we will execute together.*

### Phase 1: The Handshake (Networking)

*Goal: Establish the 200Gbps direct link.*

1.  **Physical Connection:** Connect the **QSFP56 DAC Cable** between **Port 1** of both DGX units.
2.  **Master Node (My Unit):** I will configure my IP to 192.168.100.1.
3.  **Worker Node (Your Unit):** You will run this to set your IP:

<!-- end list -->

``` bash
# Replace <INTERFACE> with the actual QSFP interface name (e.g., enP2p1s0f1np1)
sudo ip addr add 192.168.100.2/24 dev <INTERFACE>
sudo ip link set <INTERFACE> up

```

1.  **Verification:** We ping each other to ensure \< 0.1ms latency.

### Phase 2: The Validation (Speed Test)

Goal: Prove we are hitting 24GB/s bandwidth.

I will run the MPI command from the Master node to drive both GPUs:

``` bash
mpirun --allow-run-as-root -np 2 -H 192.168.100.1,192.168.100.2 \
~/SourceCodeCompiled/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

```

### Phase 3: The Visual Demo (Distributed Flux.1)

*Goal: Generate a 4K image by splitting the transformer layers across both GPUs.*

  * **Master Command:** I initiate the coordinate server.

<!-- end list -->

``` bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.100.1 ...

```

  * **Worker Command:** You join the process.

<!-- end list -->

``` bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=192.168.100.1 ...

```

### Phase 4: The "Big Brain" (Llama 3.1 405B)

*Goal: Interactive chat with a half-trillion parameter model.*

We launch the vLLM engine distributed across both nodes. The model loads into the shared memory pool (256GB), allowing us to chat with it in real-time.

``` bash
# This runs on Master and automatically controls your node via Ray
docker run --gpus all --network host --ipc=host ... --tensor-parallel-size 2

```

Cherif: note from from website <https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-10.html>

  * vllm serve uses aggressive GPU memory allocation by default (effectively `--gpu-memory-utilization`≈1.0). On systems with shared/unified GPU memory (e.g. DGX Spark or Jetson platforms), this can lead to out-of-memory errors. If you encounter OOM, start vllm serve with a lower utilization value, for example: `vllm serve <model> --gpu-memory-utilization 0.7`.

-----

## Appendix I - Physical Checklist (Things to Bring)

1.  NVIDIA DGX Spark Unit
2.  Power Cable for DGX Spark
3.  Fan for DGX
4.  QSFP56 200Gbps
5.  Laptop with charging cable
6.  Mouse for laptop
7.  USB NVME **drive**

## Appendix II - What is Llama 3.1 405B capable of?

**Summary and Capabilities of Llama 3.1 405B (AI Generated)**

Llama 3.1 405B is the largest and most powerful version in the Llama 3.1 series, designed to deliver state-of-the-art performance.

**Key Capabilities:** It offers significant performance improvements over previous Llama models, particularly excelling in complex tasks requiring deep reasoning, advanced coding, and comprehensive general knowledge. A notable feature is enhanced **multilinguality**, allowing it to understand and generate high-quality text in various languages beyond English. It also supports an industry-leading **long context window**, enabling it to process and maintain coherence over massive amounts of text or extensive conversation histories. The 405B model is specifically tuned to aim for top-tier, state-of-the-art performance across all major benchmarks.

**Comparison to Other Models:**

Llama 3.1 405B generally compares directly against the most advanced proprietary and open-source foundation models in the AI landscape, such as:

  * **Anthropic's Claude 3 Opus** (its top-tier model)
  * **OpenAI's GPT-4o and GPT-4**
  * **Google's top Gemini models**
  * **Other leading open-source models** like Mistral Large or the Qwen series.

As an open-source model, Llama 3.1 405B's primary comparison is against proprietary state-of-the-art models (like GPT-4o or Claude 3 Opus) in terms of raw performance, and against other open-source models in terms of accessibility, license, and community adoption.

**Llama 3.1 405B: Technical Specifications & Market Position**

| Feature / Aspect          | Specification & Details                                                                            | Significance                                                                                                  | Reference / Source                                                                                                     |
| :-----------------------: | :------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| Model Type                | 405 Billion Parameter Dense Transformer                                                            | The largest open-weights model to date, designed to rival top-tier proprietary systems.                       | [Meta Official Blog](https://ai.meta.com/blog/meta-llama-3-1/)                                                         |
| Context Window            | 128,000 Tokens (\~95,000 words)                                                                    | Significantly expanded from Llama 3 (8k), enabling heavy RAG applications and long-document analysis.         | [Hugging Face Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B)                                       |
| Performance               | State-of-the-art reasoning, math, and coding.                                                      | Matches or rivals GPT-4o and Claude 3.5 Sonnet on major benchmarks (GSM8K, HumanEval, MMLU).                  | [Artificial Analysis](https://www.google.com/search?q=https://artificialanalysis.ai/models/llama-3-1-405b-instruct)    |
| Multilinguality           | Supports 8 core languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. | Improved syntax and nuance for non-English generation compared to previous iterations.                        | [Meta Research Paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)                           |
| Distillation Capabilities | Authorized for Synthetic Data Generation                                                           | The license explicitly permits using 405B outputs to train/fine-tune smaller models (e.g., Llama 3.1 8B/70B). | [Meta Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)                 |
| Licensing                 | Meta Community License (Open Weights)                                                              | Free for research and commercial use (unless \>700M monthly users). *Note: Not strictly OSI Open Source.*     | [Meta License Terms](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)                     |
| Primary Competitors       | GPT-4o (OpenAI), Claude 3.5 Sonnet (Anthropic), Gemini 1.5 Pro (Google)                            | Bridges the gap between open-access models and proprietary frontier models.                                   | [Vellum.ai Benchmarks](https://www.vellum.ai/blog/evaluating-llama-3-1-405b-against-leading-closed-source-competitors) |

Key Source Descriptions

  * Meta Official Blog: The primary announcement detailing the architecture, training scale (16k H100 GPUs), and strategic goals.
  * Hugging Face Model Card: The technical documentation repository containing direct weight downloads, usage instructions, and specific token limits.
  * Meta Community License: The legal text verifying the "Open Weights" status and the specific permission grant for "Distillation" (using model outputs to train other models), which is unique among frontier models.
  * Artificial Analysis / Vellum.ai: Third-party independent benchmarking platforms that validate the performance claims against OpenAI and Anthropic models.

<!-- end list -->

``` 
 
```
