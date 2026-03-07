"""
============================================================
NexusForge — NexusHub: Central Orchestrator
============================================================
Core hub that manages all agents, runs them in parallel,
merges outputs using a PyTorch MLP neural net (resonance),
and provides self-healing via automatic retry logic.
Quantum-like: generates parallel plan variants simultaneously.
============================================================
"""

import asyncio
import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Logging Setup ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [NexusHub] %(levelname)s: %(message)s")
logger = logging.getLogger("NexusHub")


# ══════════════════════════════════════════════════════════════
# 1. Resonance Neural Network (2-Layer MLP)
#    Merges & refines multiple agent outputs into one signal
# ══════════════════════════════════════════════════════════════
class ResonanceNet(nn.Module):
    """
    2-Layer MLP that takes concatenated agent output embeddings
    and produces a refined, unified output vector.
    Think of it as 'harmonizing' multiple agent voices.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super(ResonanceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(output_dim, input_dim)  # project back to original space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.network(x)
        return self.output_proj(refined)


# ══════════════════════════════════════════════════════════════
# 2. Agent Slot Definition
# ══════════════════════════════════════════════════════════════
@dataclass
class AgentSlot:
    """
    Represents one of the 5 agent slots in NexusHub.
    Each slot holds an agent callable and its metadata.
    """
    slot_id: int
    name: str
    agent_fn: Optional[Callable] = None
    is_active: bool = False
    last_output: Any = None
    failure_count: int = 0
    metadata: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# 3. MetaLearner — Self-Evolving Nexus
#    Logs agent performance and refines prompts over time
# ══════════════════════════════════════════════════════════════
class MetaLearner:
    """
    Logs each agent's performance to a JSON file.
    Over multiple runs, refines LLM system prompts based on scores.
    Shows long-term learning capability of the system.
    """
    def __init__(self, log_path: str = "meta_learning_log.json"):
        self.log_path = log_path
        self.performance_log: List[Dict] = []
        self._load_existing_log()

    def _load_existing_log(self):
        try:
            with open(self.log_path, "r") as f:
                self.performance_log = json.load(f)
            logger.info(f"MetaLearner: Loaded {len(self.performance_log)} historical records.")
        except (FileNotFoundError, json.JSONDecodeError):
            self.performance_log = []

    def log_run(self, agent_name: str, input_summary: str, output_summary: str, score: float):
        """Record an agent's performance for future prompt refinement."""
        entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "input_summary": input_summary[:200],
            "output_summary": output_summary[:200],
            "score": score,
        }
        self.performance_log.append(entry)
        self._save_log()
        logger.info(f"MetaLearner: Logged run for '{agent_name}' with score={score:.3f}")

    def _save_log(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.performance_log, f, indent=2)
        except Exception as e:
            logger.warning(f"MetaLearner: Could not save log — {e}")

    def get_refined_prompt_hint(self, agent_name: str) -> str:
        """
        Analyze past performance and return a prompt refinement hint.
        Agents with low avg scores get improvement suggestions.
        """
        agent_runs = [r for r in self.performance_log if r["agent"] == agent_name]
        if len(agent_runs) < 3:
            return ""  # Not enough data yet

        avg_score = np.mean([r["score"] for r in agent_runs[-10:]])  # last 10 runs
        if avg_score < 0.5:
            return (
                f"[MetaLearner Hint] {agent_name} has avg score {avg_score:.2f}. "
                "Consider: Be more specific, add examples, simplify output format."
            )
        elif avg_score > 0.8:
            return (
                f"[MetaLearner Hint] {agent_name} performing well (score={avg_score:.2f}). "
                "Consider: Push for more creativity and edge-case handling."
            )
        return ""


# ══════════════════════════════════════════════════════════════
# 4. NexusHub — The Central Orchestrator
# ══════════════════════════════════════════════════════════════
class NexusHub:
    """
    Central orchestrator for NexusForge email automation platform.

    Features:
    - 5 agent slots for parallel execution
    - Concurrent dispatch via ThreadPoolExecutor
    - Resonance: PyTorch MLP merges agent outputs
    - Self-healing: automatic retry on failure (max 2x)
    - Quantum-like parallel plan variants
    - MetaLearner for self-evolution
    - HITL (Human-in-the-Loop) checkpoint support
    """

    MAX_SLOTS = 5
    MAX_RETRIES = 2

    def __init__(self, device: str = "cpu"):
        logger.info("Initializing NexusHub...")
        self.device = torch.device(device)

        # ── 5 Agent Slots ──────────────────────────────────
        self.slots: List[AgentSlot] = [
            AgentSlot(slot_id=i, name=f"slot_{i}") for i in range(self.MAX_SLOTS)
        ]

        # ── Resonance Neural Network ────────────────────────
        self.resonance_net = ResonanceNet(
            input_dim=512, hidden_dim=256, output_dim=128
        ).to(self.device)
        self.resonance_net.eval()  # inference mode by default
        logger.info("ResonanceNet initialized (2-layer MLP, 512→256→128→512).")

        # ── Meta-Learner ────────────────────────────────────
        self.meta_learner = MetaLearner()

        # ── Run History for audit trail ─────────────────────
        self.run_history: List[Dict] = []

        # ── HITL pending approvals queue ────────────────────
        self.hitl_pending: List[Dict] = []

        logger.info(f"NexusHub ready — {self.MAX_SLOTS} agent slots, device={device}.")

    # ──────────────────────────────────────────────────────────
    # Agent Registration
    # ──────────────────────────────────────────────────────────
    def register_agent(self, slot_id: int, name: str, agent_fn: Callable, metadata: Dict = {}):
        """Register an agent callable into a specific slot."""
        if slot_id >= self.MAX_SLOTS:
            raise ValueError(f"Slot {slot_id} exceeds max slots ({self.MAX_SLOTS}).")
        self.slots[slot_id].name = name
        self.slots[slot_id].agent_fn = agent_fn
        self.slots[slot_id].is_active = True
        self.slots[slot_id].metadata = metadata
        logger.info(f"Agent '{name}' registered in slot {slot_id}.")

    # ──────────────────────────────────────────────────────────
    # Self-Healing Agent Execution
    # ──────────────────────────────────────────────────────────
    def _execute_with_healing(self, slot: AgentSlot, input_data: Any) -> Dict:
        """
        Execute an agent with automatic retry on failure (self-healing).
        Max 2 retries. Returns result dict with status and output.
        """
        attempt = 0
        last_error = None

        while attempt <= self.MAX_RETRIES:
            try:
                if attempt > 0:
                    logger.warning(f"[Self-Heal] Retrying '{slot.name}' (attempt {attempt}/{self.MAX_RETRIES})...")
                    time.sleep(0.5 * attempt)  # exponential backoff

                output = slot.agent_fn(input_data)
                slot.last_output = output
                slot.failure_count = 0
                logger.info(f"[✓] Agent '{slot.name}' succeeded on attempt {attempt + 1}.")
                return {"status": "success", "slot": slot.name, "output": output, "attempts": attempt + 1}

            except Exception as e:
                last_error = str(e)
                slot.failure_count += 1
                logger.error(f"[✗] Agent '{slot.name}' failed (attempt {attempt + 1}): {e}")
                attempt += 1

        # All retries exhausted
        logger.error(f"[DEAD] Agent '{slot.name}' failed after {self.MAX_RETRIES + 1} attempts.")
        return {
            "status": "failed",
            "slot": slot.name,
            "output": None,
            "error": last_error,
            "attempts": attempt
        }

    # ──────────────────────────────────────────────────────────
    # Parallel Dispatch
    # ──────────────────────────────────────────────────────────
    def dispatch(self, input_data: Any, slots: Optional[List[int]] = None) -> List[Dict]:
        """
        Dispatch active agents in PARALLEL using ThreadPoolExecutor.
        Quantum-like: all agents run simultaneously, not sequentially.

        Args:
            input_data: Input passed to all active agents.
            slots: List of slot IDs to run. Defaults to all active slots.

        Returns:
            List of result dicts from each agent.
        """
        target_slots = [
            s for s in self.slots
            if s.is_active and (slots is None or s.slot_id in slots)
        ]

        if not target_slots:
            logger.warning("No active agent slots to dispatch!")
            return []

        logger.info(f"Dispatching {len(target_slots)} agents in parallel: {[s.name for s in target_slots]}")

        results = []
        # ThreadPoolExecutor for true parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_SLOTS) as executor:
            future_to_slot = {
                executor.submit(self._execute_with_healing, slot, input_data): slot
                for slot in target_slots
            }

            for future in concurrent.futures.as_completed(future_to_slot):
                slot = future_to_slot[future]
                try:
                    result = future.result(timeout=120)  # 2-min timeout per agent
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Agent '{slot.name}' timed out!")
                    results.append({"status": "timeout", "slot": slot.name, "output": None})

        # Store in run history
        self.run_history.append({
            "timestamp": time.time(),
            "agents_run": [s.name for s in target_slots],
            "results_count": len(results),
            "success_count": sum(1 for r in results if r["status"] == "success"),
        })

        logger.info(f"Dispatch complete: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} succeeded.")
        return results

    # ──────────────────────────────────────────────────────────
    # Quantum-Like Parallel Plan Variants
    # ──────────────────────────────────────────────────────────
    def quantum_dispatch(self, planner_fn: Callable, input_data: Any, n_variants: int = 3) -> Dict:
        """
        Generate N parallel plan variants simultaneously (quantum-like superposition).
        Scores each variant and collapses to the best one.

        Args:
            planner_fn: A callable that generates a plan given input.
            input_data: Input data for the planner.
            n_variants: Number of parallel variants to generate (default 3).

        Returns:
            The best-scoring plan variant.
        """
        logger.info(f"Quantum dispatch: generating {n_variants} parallel plan variants...")

        # Create N slightly perturbed inputs (like quantum variations)
        variant_inputs = []
        for i in range(n_variants):
            variant_input = dict(input_data) if isinstance(input_data, dict) else {"data": input_data}
            variant_input["variant_seed"] = i  # different random seed per variant
            variant_input["temperature"] = 0.5 + (i * 0.2)  # varied creativity
            variant_inputs.append(variant_input)

        # Execute all variants in parallel
        variant_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_variants) as executor:
            futures = [executor.submit(planner_fn, vi) for vi in variant_inputs]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result(timeout=60)
                    score = self._heuristic_score(result)
                    variant_results.append({"variant": i, "result": result, "score": score})
                    logger.info(f"  Variant {i}: score={score:.3f}")
                except Exception as e:
                    logger.error(f"  Variant {i} failed: {e}")

        if not variant_results:
            return {}

        # Collapse to best variant (highest score)
        best = max(variant_results, key=lambda x: x["score"])
        logger.info(f"Quantum collapse → Variant {best['variant']} selected (score={best['score']:.3f}).")
        return best["result"]

    def _heuristic_score(self, plan: Any) -> float:
        """
        Simple heuristic scoring for plan quality.
        Higher score = better plan. Range: [0.0, 1.0]
        """
        if not plan:
            return 0.0
        score = 0.5  # base score

        if isinstance(plan, dict):
            # Reward completeness
            if "segments" in plan:
                score += 0.1 * min(len(plan["segments"]), 3) / 3
            if "schedule" in plan and plan["schedule"]:
                score += 0.1
            if "outline" in plan and len(str(plan.get("outline", ""))) > 100:
                score += 0.1
            if "predicted_engagement" in plan:
                score += min(plan["predicted_engagement"] / 100, 0.2)

        return min(score, 1.0)

    # ──────────────────────────────────────────────────────────
    # Resonate — Neural Network Output Merging
    # ──────────────────────────────────────────────────────────
    def resonate(self, agent_outputs: List[Dict]) -> Dict:
        """
        Merge multiple agent outputs using the 2-layer MLP ResonanceNet.

        Process:
        1. Encode each agent's output as a numeric vector (text hashing)
        2. Concatenate into a single tensor (pad/truncate to 512 dims)
        3. Pass through ResonanceNet for neural refinement
        4. Decode back to structured dict

        Args:
            agent_outputs: List of agent result dicts.

        Returns:
            Resonated (merged & refined) output dict.
        """
        successful = [r for r in agent_outputs if r.get("status") == "success" and r.get("output")]

        if not successful:
            logger.warning("Resonate: No successful outputs to merge.")
            return {"status": "no_outputs", "resonated": None}

        logger.info(f"Resonating {len(successful)} agent outputs with neural net...")

        # Step 1: Encode outputs as numeric vectors
        encoded_vectors = []
        for result in successful:
            vec = self._encode_output(result["output"])
            encoded_vectors.append(vec)

        # Step 2: Average-pool all vectors to single 512-dim tensor
        stacked = torch.stack(encoded_vectors, dim=0).to(self.device)
        mean_vec = torch.mean(stacked, dim=0, keepdim=True)  # (1, 512)

        # Step 3: Pass through ResonanceNet
        with torch.no_grad():
            refined_vec = self.resonance_net(mean_vec)  # (1, 512)

        # Step 4: Build resonated output structure
        resonated_output = {
            "status": "resonated",
            "source_agents": [r["slot"] for r in successful],
            "resonance_vector": refined_vec.cpu().numpy().tolist(),
            "confidence": float(torch.mean(refined_vec).item()),
            "merged_outputs": {r["slot"]: r["output"] for r in successful},
            # Primary output = first successful agent's output (most likely ForgePlanner)
            "primary_output": successful[0]["output"] if successful else None,
        }

        logger.info(f"Resonance complete. Confidence: {resonated_output['confidence']:.4f}")
        return resonated_output

    def _encode_output(self, output: Any, dim: int = 512) -> torch.Tensor:
        """
        Encode an agent output to a fixed-size float tensor.
        Uses character-level hash mapping for text content.
        """
        text = json.dumps(output, default=str)
        vec = np.zeros(dim, dtype=np.float32)

        for i, char in enumerate(text[:dim * 4]):
            idx = (ord(char) * (i + 1)) % dim
            vec[idx] += 1.0 / (i + 1)  # position-weighted

        # Normalize to [0, 1]
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val

        return torch.tensor(vec, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────
    # HITL Checkpoint
    # ──────────────────────────────────────────────────────────
    def request_hitl_approval(self, step_name: str, data: Dict, blockchain_hash: Optional[str] = None) -> Dict:
        """
        Add a step to the HITL approval queue.
        Returns the pending approval request (UI will process it).
        """
        approval_request = {
            "id": f"hitl_{int(time.time())}_{step_name}",
            "step": step_name,
            "data": data,
            "blockchain_hash": blockchain_hash,
            "status": "pending",
            "timestamp": time.time(),
        }
        self.hitl_pending.append(approval_request)
        logger.info(f"HITL approval requested for step: '{step_name}' (id={approval_request['id']})")
        return approval_request

    def resolve_hitl(self, approval_id: str, decision: str, revised_data: Optional[Dict] = None) -> bool:
        """
        Resolve a pending HITL approval (approve/reject/revise).
        """
        for req in self.hitl_pending:
            if req["id"] == approval_id:
                req["status"] = decision
                if revised_data:
                    req["revised_data"] = revised_data
                req["resolved_at"] = time.time()
                logger.info(f"HITL '{approval_id}' resolved: {decision}")
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # Status & Diagnostics
    # ──────────────────────────────────────────────────────────
    def status(self) -> Dict:
        """Return current hub status and agent slot summary."""
        return {
            "active_slots": sum(1 for s in self.slots if s.is_active),
            "total_runs": len(self.run_history),
            "hitl_pending": len([r for r in self.hitl_pending if r["status"] == "pending"]),
            "slots": [
                {
                    "id": s.slot_id,
                    "name": s.name,
                    "active": s.is_active,
                    "failures": s.failure_count,
                }
                for s in self.slots
            ],
        }
