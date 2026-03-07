# ⚡ NexusForge — AI-Powered BFSI Email Automation Platform

> **Agentic AI system for intelligent, compliant, and personalized BFSI email campaigns**

---

## 🏗️ Architecture Overview

```
NexusForge/
├── core/
│   ├── nexus_hub.py          ← Central Orchestrator (NexusHub + MetaLearner)
│   └── hitl_blockchain.py    ← Human-in-the-Loop + Ethereum Blockchain Logger
├── agents/
│   ├── forge_planner.py      ← ForgePlanner: Customer segmentation + 3 parallel plans
│   ├── forge_content_gen.py  ← ForgeContentGen: Jinja2 HTML + Matplotlib charts
│   └── forge_agents.py       ← ForgeScheduler + ForgeAnalyzer + ForgeOptimizer
├── ml/
│   └── foresight.py          ← ForesightOracle (Prophet) + ComplianceGuard
├── utils/
│   └── ar_viz.py             ← ARViz: QR code + AR.js 3D email preview
├── data/
│   └── generate_data.py      ← 100 fake BFSI customers + historical opens CSV
├── app.py                    ← Streamlit Dashboard (6 pages)
└── requirements.txt
```

---

## 🤖 Agents & Features

| Agent | Description | Tech |
|-------|-------------|------|
| **NexusHub** | Central orchestrator, parallel dispatch, neural resonance | PyTorch MLP, concurrent.futures |
| **ForgePlanner** | Segment customers, generate 3 parallel plans, select best | LangChain, Prophet foresight |
| **ForgeContentGen** | Personalized HTML emails + dynamic charts | Jinja2, Matplotlib, VADER |
| **ForgeScheduler** | Batch send with A/B testing + tracking | SendGrid, Celery, asyncio |
| **ForgeAnalyzer** | Metrics + z-score anomaly detection | Pandas, SciPy, Plotly |
| **ForgeOptimizer** | RL-based campaign refinement (Q-learning) | NumPy Q-learning |
| **ComplianceGuard** | PII masking + BFSI compliance scan | Regex, VADER sentiment |
| **ForesightOracle** | Time-series demand forecasting | Facebook Prophet |
| **HITLBlockchain** | Human approval + Ethereum audit trail | Web3.py, Sepolia testnet |
| **VoiceApprove** | Voice-activated HITL via Whisper | OpenAI Whisper |
| **ARViz** | QR code → AR.js 3D email preview | AR.js, A-Frame, qrcode |
| **MetaLearner** | Self-evolving prompts from run history | JSON logging, NumPy |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Demo Data
```bash
python data/generate_data.py
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. (Optional) Set API Keys
```bash
export OPENAI_API_KEY="sk-..."          # For LLM-powered planning
export SENDGRID_API_KEY="SG...."        # For real email sending
export ETHEREUM_PRIVATE_KEY="0x..."     # For real blockchain logging
```

---

## ⚡ Key Technical Highlights

### Quantum-Like Parallel Execution
```python
hub = NexusHub(device="cpu")
hub.register_agent(0, "planner", planner_fn)
results = hub.dispatch(input_data)  # All agents run simultaneously
```

### Neural Resonance (PyTorch MLP)
```python
resonated = hub.resonate(agent_outputs)
# 2-layer MLP: 512 → 256 → 128 → 512
# Merges all agent outputs into one refined signal
```

### Self-Healing
```python
# Agents auto-retry on failure (max 2x with exponential backoff)
result = hub._execute_with_healing(slot, input_data)
```

### Blockchain Audit Trail
```python
blockchain = BlockchainLogger()
tx = blockchain.log_approval("plan_approved", plan_data, "approved")
# Returns: {"tx_hash": "0x...", "block_number": 5800001, "network": "Sepolia"}
```

### BFSI Compliance Check
```python
guard = ComplianceGuard()
result = guard.check(email_html, "loan_offer")
# Masks Aadhaar, PAN, phones; checks SEBI disclaimers; sentiment scan
```

---

## 👥 Team

> **Add your team details in the app:**
> Navigate to `👥 Team` page in the Streamlit dashboard to add:
> - Team name
> - Member names and roles
> - Hackathon/event name
> - Institution

---

## 🎯 BFSI Use Cases

- **Home/Personal Loan** campaigns with personalized EMI charts
- **Mutual Fund SIP** campaigns with projected wealth growth
- **Credit Card** upgrade campaigns with benefit breakdowns
- **Insurance** campaigns with coverage comparison tables
- **Fixed Deposit** campaigns with maturity calculators

---

## 📋 Demo Flow (No Real API Keys Needed)

1. Open `http://localhost:8501`
2. Go to **Create Campaign**
3. Use demo CSV or upload your own
4. Click **Run NexusForge Campaign**
5. View results in **Analytics**
6. Approve in **Approvals** (blockchain simulated)
7. Chat with AI in **AI Chat**

---

## 🔮 Futuristic Features

- **Quantum dispatch**: N parallel plan variants evaluated simultaneously
- **MetaLearner**: Improves agent prompts from historical performance
- **AR Preview**: 3D email visualization via phone camera
- **Voice HITL**: "Yes" / "No" voice commands for approvals
- **Blockchain audit**: Immutable compliance trail on Ethereum
- **Foresight Oracle**: "15% drop if high inflation" predictions

---

*Built with ❤️ for BFSI innovation | NexusForge Platform*
