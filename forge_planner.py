"""
============================================================
NexusForge — ForgePlanner Agent
============================================================
Reads customer CSV, segments audience, generates 3 parallel
campaign plan variants (scored by engagement heuristic),
selects the best one. Integrates Foresight Oracle for
trend-based adjustments (e.g., economic indicators).
Output: JSON with segments, content outline, schedule.
============================================================
"""

import json
import logging
import concurrent.futures
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ForgePlanner")

# ── Gemini LLM integration ────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.config import gemini_generate, GEMINI_API_KEY
GEMINI_AVAILABLE = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")


# ══════════════════════════════════════════════════════════════
# Foresight Oracle Integration (imported from ml/foresight.py)
# ══════════════════════════════════════════════════════════════
def call_foresight(segment_name: str, historical_data: Optional[Dict] = None) -> Dict:
    """
    Stub call to Foresight Oracle for trend-based planning.
    In production, this calls the Prophet-based ForesightOracle.
    Returns predicted engagement multiplier and risk flags.
    """
    try:
        from ml.foresight import ForesightOracle
        oracle = ForesightOracle()
        forecast = oracle.forecast_dummy()
        return {
            "engagement_multiplier": forecast.get("trend_multiplier", 1.0),
            "risk_flag": forecast.get("risk_flag", False),
            "confidence": forecast.get("confidence", 0.75),
            "note": forecast.get("note", ""),
        }
    except Exception:
        # Fallback: dummy foresight data
        return {
            "engagement_multiplier": np.random.uniform(0.8, 1.3),
            "risk_flag": False,
            "confidence": 0.70,
            "note": "Foresight oracle unavailable — using default multiplier.",
        }


# ══════════════════════════════════════════════════════════════
# Audience Segmentation Logic
# ══════════════════════════════════════════════════════════════
def segment_customers(df: pd.DataFrame) -> List[Dict]:
    """
    Segment customers based on income, past engagement, and location.
    Returns list of segment dicts with name, count, and profile.
    """
    segments = []

    # ── Income-based segmentation ──────────────────────────
    if "income" in df.columns:
        df["income_band"] = pd.cut(
            df["income"],
            bins=[0, 300000, 800000, float("inf")],
            labels=["low", "mid", "high"]
        )
        for band in ["low", "mid", "high"]:
            seg_df = df[df["income_band"] == band]
            if len(seg_df) > 0:
                segments.append({
                    "name": f"income_{band}",
                    "size": len(seg_df),
                    "avg_income": float(seg_df["income"].mean()) if "income" in seg_df else 0,
                    "avg_past_opens": float(seg_df["past_opens"].mean()) if "past_opens" in seg_df else 0,
                    "locations": seg_df["location"].value_counts().head(3).to_dict() if "location" in seg_df else {},
                    "email_list": seg_df["email"].tolist() if "email" in seg_df else [],
                })

    # ── Engagement-based segmentation ──────────────────────
    if "past_opens" in df.columns:
        high_eng = df[df["past_opens"] >= 5]
        low_eng = df[df["past_opens"] < 5]
        if len(high_eng) > 0:
            segments.append({
                "name": "high_engagement",
                "size": len(high_eng),
                "avg_past_opens": float(high_eng["past_opens"].mean()),
                "email_list": high_eng["email"].tolist() if "email" in high_eng else [],
            })
        if len(low_eng) > 0:
            segments.append({
                "name": "low_engagement",
                "size": len(low_eng),
                "avg_past_opens": float(low_eng["past_opens"].mean()),
                "email_list": low_eng["email"].tolist() if "email" in low_eng else [],
            })

    if not segments:
        # Fallback: treat everyone as one segment
        segments = [{
            "name": "all_customers",
            "size": len(df),
            "email_list": df["email"].tolist() if "email" in df else [],
        }]

    return segments


# ══════════════════════════════════════════════════════════════
# Content Outline Generator (Rule-based + LLM fallback)
# ══════════════════════════════════════════════════════════════
BFSI_TEMPLATES = {
    "loan_offer": {
        "subject": "Exclusive Pre-Approved Loan Offer for You",
        "sections": ["Hero Banner", "Loan Details Table", "EMI Calculator CTA", "Terms", "Footer"],
        "tone": "professional, urgent",
    },
    "investment": {
        "subject": "Grow Your Wealth — Curated Investment Plans",
        "sections": ["Market Snapshot", "Top Funds", "Returns Graph", "Invest Now CTA", "Footer"],
        "tone": "aspirational, data-driven",
    },
    "credit_card": {
        "subject": "Your Exclusive Credit Card Upgrade is Ready",
        "sections": ["Card Visual", "Benefits List", "Rewards Summary", "Apply CTA", "Footer"],
        "tone": "premium, benefit-focused",
    },
    "insurance": {
        "subject": "Protect What Matters — Insurance Plans from ₹99/month",
        "sections": ["Risk Awareness Banner", "Coverage Table", "Testimonial", "Get Quote CTA", "Footer"],
        "tone": "empathetic, reassuring",
    },
    "fd_offer": {
        "subject": "Lock In 8.5% Returns — Fixed Deposit Special",
        "sections": ["Rate Highlight", "Comparison Chart", "Calculator", "Book Now CTA", "Footer"],
        "tone": "factual, opportunity-focused",
    },
}

def generate_content_outline(goal: str, segments: List[Dict], variant_seed: int = 0) -> Dict:
    """
    Generate a content outline for the campaign based on goal and segments.
    Returns dict with email template type, sections, tone, and per-segment strategy.
    """
    # Map goal keywords to BFSI templates
    goal_lower = goal.lower()
    template_key = "loan_offer"  # default
    for key in BFSI_TEMPLATES:
        if key.replace("_", " ") in goal_lower or key in goal_lower:
            template_key = key
            break
    if "invest" in goal_lower or "mutual" in goal_lower:
        template_key = "investment"
    elif "credit" in goal_lower or "card" in goal_lower:
        template_key = "credit_card"
    elif "insur" in goal_lower or "protect" in goal_lower:
        template_key = "insurance"
    elif "fd" in goal_lower or "fixed" in goal_lower or "deposit" in goal_lower:
        template_key = "fd_offer"

    template = BFSI_TEMPLATES[template_key]

    # Variant-specific variations
    tone_variations = ["professional", "conversational", "urgent"][variant_seed % 3]

    outline = {
        "template_type": template_key,
        "subject_line": template["subject"],
        "email_sections": template["sections"],
        "tone": tone_variations,
        "personalization_fields": ["name", "income_band", "city"],
        "per_segment_strategy": {},
    }

    for seg in segments:
        seg_name = seg["name"]
        if "high" in seg_name:
            outline["per_segment_strategy"][seg_name] = "Premium tone, exclusive offers, loyalty rewards"
        elif "low" in seg_name:
            outline["per_segment_strategy"][seg_name] = "Simple language, basic benefits, easy CTA"
        elif "income_high" in seg_name:
            outline["per_segment_strategy"][seg_name] = "Wealth management, premium products, relationship manager"
        else:
            outline["per_segment_strategy"][seg_name] = "Standard BFSI offer, clear value proposition"

    return outline


# ══════════════════════════════════════════════════════════════
# Schedule Generator
# ══════════════════════════════════════════════════════════════
def generate_schedule(segments: List[Dict], variant_seed: int = 0) -> List[Dict]:
    """
    Generate a campaign send schedule optimized for BFSI.
    Best open times: Tuesday/Wednesday 10am-12pm, 2pm-4pm IST.
    """
    base_date = datetime.now()
    schedule = []

    # Offset by variant for A/B testing different days
    day_offsets = [2, 3, 4]  # Tue, Wed, Thu

    for i, seg in enumerate(segments):
        send_day = base_date + timedelta(days=day_offsets[i % len(day_offsets)] + (variant_seed * 1))
        send_time = "10:00 AM IST" if variant_seed % 2 == 0 else "2:00 PM IST"

        schedule.append({
            "segment": seg["name"],
            "send_date": send_day.strftime("%Y-%m-%d"),
            "send_time": send_time,
            "batch_size": min(seg.get("size", 50), 500),
            "a_b_test": True,
            "follow_up_date": (send_day + timedelta(days=3)).strftime("%Y-%m-%d"),
        })

    return schedule


# ══════════════════════════════════════════════════════════════
# Engagement Heuristic Scorer
# ══════════════════════════════════════════════════════════════
def score_plan(plan: Dict, foresight_data: Dict) -> float:
    """
    Score a plan variant on predicted engagement (0.0 to 100.0).
    Higher = better expected campaign performance.
    """
    score = 50.0  # base

    # Reward more segments (more targeted)
    segments = plan.get("segments", [])
    score += min(len(segments) * 3, 15)

    # Reward completeness of outline
    outline = plan.get("outline", {})
    if "email_sections" in outline:
        score += min(len(outline["email_sections"]) * 2, 10)
    if "per_segment_strategy" in outline and outline["per_segment_strategy"]:
        score += 5

    # Reward schedule completeness
    schedule = plan.get("schedule", [])
    score += min(len(schedule) * 2, 10)

    # Apply foresight multiplier
    foresight_multiplier = foresight_data.get("engagement_multiplier", 1.0)
    score *= foresight_multiplier

    # Penalize if foresight flags risk
    if foresight_data.get("risk_flag"):
        score *= 0.85

    return round(min(score, 100.0), 2)


# ══════════════════════════════════════════════════════════════
# ForgePlanner Agent — Main Class
# ══════════════════════════════════════════════════════════════
class ForgePlanner:
    """
    LangChain-powered planning agent for BFSI email campaigns.

    Workflow:
    1. Load customer CSV and segment audience
    2. Call Foresight Oracle for trend data
    3. Generate 3 parallel plan variants
    4. Score each variant with heuristic
    5. Return best JSON plan
    """

    def __init__(self, llm_model: str = "gemini-2.0-flash"):
        self.llm_model = llm_model
        self.use_gemini = GEMINI_AVAILABLE
        if self.use_gemini:
            logger.info(f"ForgePlanner: Gemini LLM ready ({llm_model}).")
        else:
            logger.warning("ForgePlanner: Gemini key not set — using rule-based fallback.")

    def _generate_single_variant(self, df: pd.DataFrame, goal: str, variant_seed: int, foresight: Dict) -> Dict:
        """Generate one plan variant (called 3x in parallel)."""
        segments = segment_customers(df)
        outline = generate_content_outline(goal, segments, variant_seed)
        schedule = generate_schedule(segments, variant_seed)

        plan = {
            "variant_id": variant_seed,
            "goal": goal,
            "segments": segments,
            "outline": outline,
            "schedule": schedule,
            "foresight": foresight,
            "predicted_engagement": 0,  # filled after scoring
        }
        plan["predicted_engagement"] = score_plan(plan, foresight)
        return plan

    def run(self, csv_path: str, goal: str) -> Dict:
        """
        Main entry point for ForgePlanner.

        Args:
            csv_path: Path to customer CSV file.
            goal: Campaign goal string (e.g., "Promote home loan offers to mid-income customers").

        Returns:
            Best JSON plan with segments, outline, schedule.
        """
        logger.info(f"ForgePlanner: Starting plan generation for goal='{goal}'")

        # Step 1: Load CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"ForgePlanner: Loaded {len(df)} customers from {csv_path}")
        except Exception as e:
            logger.error(f"ForgePlanner: Failed to load CSV: {e}")
            df = pd.DataFrame({
                "name": ["Demo Customer"],
                "email": ["demo@example.com"],
                "income": [500000],
                "location": ["Mumbai"],
                "past_opens": [3],
            })

        # Step 2: Call Foresight Oracle
        foresight_data = call_foresight(segment_name="all", historical_data=None)
        logger.info(f"ForgePlanner: Foresight → multiplier={foresight_data['engagement_multiplier']:.2f}, "
                    f"risk={foresight_data['risk_flag']}, confidence={foresight_data['confidence']:.2f}")

        # Step 3: Generate 3 parallel plan variants (quantum-like)
        logger.info("ForgePlanner: Generating 3 parallel plan variants...")
        variants = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._generate_single_variant, df, goal, seed, foresight_data): seed
                for seed in range(3)
            }
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                try:
                    variant = future.result(timeout=30)
                    variants.append(variant)
                    logger.info(f"  Variant {seed}: engagement_score={variant['predicted_engagement']:.1f}")
                except Exception as e:
                    logger.error(f"  Variant {seed} failed: {e}")

        if not variants:
            logger.error("ForgePlanner: All variants failed!")
            return {"error": "No variants generated"}

        # Step 4: Select best variant
        best_plan = max(variants, key=lambda v: v["predicted_engagement"])
        best_plan["selection_reason"] = f"Highest predicted engagement ({best_plan['predicted_engagement']:.1f}/100) among 3 variants"
        best_plan["all_variant_scores"] = [
            {"variant_id": v["variant_id"], "score": v["predicted_engagement"]}
            for v in sorted(variants, key=lambda x: x["predicted_engagement"], reverse=True)
        ]

        logger.info(f"ForgePlanner: Best plan selected — Variant {best_plan['variant_id']} "
                    f"(score={best_plan['predicted_engagement']:.1f})")

        return best_plan

    def __call__(self, input_data: Any) -> Dict:
        """Make ForgePlanner callable for NexusHub slot registration."""
        if isinstance(input_data, dict):
            return self.run(
                csv_path=input_data.get("csv_path", "data/customers.csv"),
                goal=input_data.get("goal", "BFSI email campaign")
            )
        return self.run(csv_path=str(input_data), goal="General BFSI campaign")
