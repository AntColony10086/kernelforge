# AI Infra competitions shortlist - checked 2026-05-22

## Rush recommendation

Primary rush choice: **DevNetwork [AI + ML] Hackathon 2026 - TrueFoundry: Resilient Agents track**.

Reason: it closes on **2026-05-28 10:00 PDT** and the TrueFoundry challenge is directly about AI infrastructure resilience: how an agent behaves when an MCP server or LLM provider errors, browns out, or goes down. This is easier to turn into a compact, demoable resume project than a long leaderboard race.

Recommended build idea for the rush entry: **ResilientOps Agent**.

- Build a small multi-agent or tool-using agent behind an LLM gateway.
- Add fallback provider routing, retries, circuit breakers, structured degradation messages, trace IDs, and failure dashboards/logs.
- Demo chaos cases: MCP server timeout, LLM 5xx/brownout, malformed tool output, partial recovery.
- Resume bullet target: "Built a resilient agent runtime with LLM gateway failover, MCP tool health checks, circuit breakers, and observability traces; submitted to DevNetwork AI/ML Hackathon TrueFoundry Resilient Agents track."

Fallbacks:

- If you want the strongest brand and can spend about 3 weeks: **Google Cloud Rapid Agent Hackathon**, preferably Dynatrace/Elastic/Arize track. Important: official rules exclude residents of China, so use this only if your contest residence is eligible.
- If you want a very infra/SRE-oriented online entry with a 3-week runway: **Splunk Agentic Ops Hackathon**, Observability or Platform & Developer Experience track.
- If you want the purest inference-performance AI Infra competition and can accept a later cutoff: **2026 Baidu CTI**, generative recommendation/ad ranking inference performance optimization.

## Near-deadline shortlist

| Priority | Competition | Deadline | Fit for AI Infra resume | Best sprint angle | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | DevNetwork [AI + ML] Hackathon 2026 - TrueFoundry: Resilient Agents | 2026-05-28 10:00 PDT | High: LLM gateway, MCP failure handling, agent reliability | Build resilient agent runtime with failover, retries, circuit breakers, logs, and failure demo | https://devnetwork-ai-ml-hack-2026.devpost.com/ |
| 2 | MinerU Data Intelligence and Frontier Corpus Challenge - Data Agent track | 2026-05-24 | Medium: document parsing/data infra/agent pipeline | Only worth it if starting from an existing PDF/RAG/data-agent prototype | https://mineru.net/MDIC2026 |
| 3 | Google Cloud Rapid Agent Hackathon | 2026-06-11 14:00 PDT | High if choosing Dynatrace/Elastic/Arize/GitLab/MongoDB track | Build an ops/debugging/developer-workflow agent using Gemini + MCP partner server | https://rapid-agent.devpost.com/ |
| 4 | DeveloperWeek New York 2026 Hackathon - Tower Pipeline Challenge | 2026-06-10 | Medium-high: data infrastructure for AI | Build a data-to-AI pipeline with serverless compute, lakehouse storage, and agent workflow | https://dwny-2026-hackathon.devpost.com/ |
| 5 | Splunk Agentic Ops Hackathon | 2026-06-15 09:00 PDT | High: observability, SecOps, platform/dev experience | Build incident triage/root-cause agent over logs/metrics/traces with Splunk AI/MCP | https://splunk.devpost.com/ |
| 6 | FIND EVIL! | 2026-06-15 23:45 EDT | High but security-heavy: agentic incident response infra | Extend Protocol SIFT with custom MCP wrappers, self-correction, traceable evidence logs | https://findevil.devpost.com/ |
| 7 | Baidu CTI 2026 | 2026-06-26 11:59 Beijing | Very high: inference performance optimization | Profile and optimize generative recommendation/ad ranking inference | https://cti.baidu.com/ |

## Active candidates

| Priority | Competition | Fit | Key dates | Team / entry | Notes | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baidu CTI 2026 - Generative recommendation ad ranking inference performance optimization | Very high: inference performance, framework/algorithm/HPC optimization | Registration/recruitment: 2026-05-06 to 2026-06-26 11:59 Beijing; summer camp in July | Solo or team, max 3 | Requires legal, real-name verified Baidu AI Studio account by 2026-06-26. During recruitment, score submissions can improve invite chance. | https://cti.baidu.com/ |
| 2 | CCF Open Source Innovation Competition - MindSpore AI/Agent NPU operator auto-generation | Very high: NPU operator generation and optimization | Registration/prelim: now to 2026-07-10; semifinal: 2026-07-11 to 2026-08-06; final: 2026-08-09 to 2026-08-10 | Submit project proposal by email; later code/docs/PPT via GitLink | Good if the plan is to work on operators/compiler/agentic code generation. Compute can be requested after a basic proposal. | https://www.mindspore.cn/activities/zh/2026-3-26 |
| 3 | CCF Open Source Innovation Competition - MetaX domestic open GPU AI innovation ecosystem track | High if choosing open-source infra task | Registration/development: now to 2026-07-10; final optimization: 2026-07-18 to 2026-08-06 | 1-5 people, optional advisor | Three task types: open-source software engineering for TileLang/vLLM/SGLang on MetaX GPU, AI Skill/MCP app development, AIGC/agent development. | https://www.elecfans.com/d/7801768.html |
| 4 | DAC / ICLAD GenAI Chip Hackathon | Medium-high: GenAI for chip design / EDA flow | Registration opened 2026-05-14; pre-DAC online phase 2026-05-25 to 2026-06-25; in-person competition 2026-07-26 | Requires DAC or I LOVE DAC registration and in-person attendance in Long Beach | Good US-side option, but travel/registration overhead makes it less convenient. | https://dac.com/2026/dac-pull-down-menu-tab-genai-chip-hackathon |
| 5 | MinerU Data Intelligence and Frontier Corpus Challenge | Medium: data/agent/RAG infrastructure, not core systems performance | Registration and submission until 2026-05-24; finals in June; awards at WAIC 2026 | Team can span organizations; choose one problem | Deadline is too close for a serious new entry unless you already have a document parsing/RAG/Data Agent prototype. | https://mineru.net/MDIC2026 |
| 6 | 2nd MLC-SLM Challenge 2026 | Medium: Speech LLM systems/evaluation, less general AI Infra | Registration began 2026-03-30; leaderboard opens 2026-06-15; paper deadline 2026-07-10 | Academic/industry/individual teams | Strong if speech LLM is acceptable; not a pure infra optimization contest. | https://www.nexdata.ai/competition/mlc-slm |
| 7 | PaddleOCR Global Derivative Model Challenge | Low-medium: model fine-tuning and dataset construction, not infra | Preliminary submissions 2026-04-01 to 2026-06-28; finals 2026-07-22 to 2026-07-24 | GitHub issue comment sign-up; individual/team | Easy to enter, but mostly OCR model adaptation and open-source contribution. | https://github.com/PaddlePaddle/PaddleOCR/issues/17858 |

## Not selected / missed

These looked relevant but are already closed as of 2026-05-22:

- 2026 京津冀（廊坊）算力算法大赛: deadline 2026-05-17.
- FlagOS Open Computing Global Challenge: registration listed as 2026-01-09 to 2026-05-20.
- FPL 2026 Agentic FPGA Backend Optimization Competition: mandatory registration deadline extended to 2026-04-03.
- ICME 2026 Low-Bit-width Large Model Quantization Challenge: registration 2026-02-10 to 2026-04-10.
- AMD E2E Model Speedrun: preselection/registration deadline reported as 2026-04-07.

## Registration prep for Baidu CTI

Needed from you before final submission:

- Baidu AI Studio account access and real-name verification.
- Solo or team decision. Team size cannot exceed 3.
- Team name.
- School/company/role information for each member.
- Contact email/phone used for the platform.
- Agreement to competition rules and data confidentiality terms.

Suggested first technical direction after registration:

- Reproduce baseline and inspect scoring metric.
- Profile the inference path before changing model quality.
- Try three lanes in parallel: batching/cache and sequence handling, quantization/precision strategy, and kernel/framework bottlenecks.
- Keep every leaderboard submission tied to a git commit and experiment note.
