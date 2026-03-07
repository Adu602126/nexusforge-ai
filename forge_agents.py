"""
============================================================
NexusForge — ForgeScheduler Agent
============================================================
Sends emails in async batches via SendGrid API.
Supports A/B testing, tracking pixels, Celery async stubs.
Output: sent IDs list with status.
============================================================
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger_scheduler = logging.getLogger("ForgeScheduler")

# ── SendGrid (graceful import) ─────────────────────────────
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, TrackingSettings, ClickTracking, OpenTracking
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Celery Stub (for async scalability with 1000+ emails)
# ══════════════════════════════════════════════════════════════
class CeleryStub:
    """
    Stub for Celery task queue integration.
    In production: configure with Redis broker for 1k+ email bursts.
    """
    def __init__(self):
        self.tasks = []

    def delay(self, fn, *args, **kwargs):
        """Simulate async task dispatch."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks.append({"id": task_id, "fn": fn.__name__, "args": args})
        logger_scheduler.info(f"  [Celery] Task {task_id} queued: {fn.__name__}")
        return task_id

    def get_pending_count(self) -> int:
        return len(self.tasks)

celery_stub = CeleryStub()


# ══════════════════════════════════════════════════════════════
# Tracking Pixel Generator
# ══════════════════════════════════════════════════════════════
def generate_tracking_pixel(email_id: str, recipient_email: str, campaign_id: str) -> str:
    """
    Generate a 1x1 tracking pixel HTML snippet.
    The pixel URL would be your tracking server endpoint.
    """
    pixel_token = hashlib.md5(f"{email_id}{recipient_email}{campaign_id}".encode()).hexdigest()[:16]
    pixel_url = f"https://track.nexusforge.io/px/{campaign_id}/{pixel_token}.gif"
    return f'<img src="{pixel_url}" width="1" height="1" border="0" alt="" style="display:none;">'


# ══════════════════════════════════════════════════════════════
# ForgeScheduler Agent
# ══════════════════════════════════════════════════════════════
class ForgeScheduler:
    """
    Sends emails in batches with A/B testing and tracking.
    Uses async batching for scalability.
    """

    BATCH_SIZE = 50
    BATCH_DELAY = 0.1  # seconds between batches

    def __init__(self, sendgrid_api_key: Optional[str] = None, from_email: str = "noreply@nexusforge.io"):
        self.api_key = sendgrid_api_key or os.getenv("SENDGRID_API_KEY", "DEMO_KEY")
        self.from_email = from_email
        self.sg_client = None
        if SENDGRID_AVAILABLE and self.api_key != "DEMO_KEY":
            self.sg_client = sendgrid.SendGridAPIClient(api_key=self.api_key)
        logger_scheduler.info(f"ForgeScheduler initialized. SendGrid: {'LIVE' if self.sg_client else 'DEMO MODE'}.")

    def _send_single(self, email_data: Dict, campaign_id: str, variant: str = "A") -> Dict:
        """
        Send a single email (or simulate in demo mode).
        Adds tracking pixel to HTML body.
        """
        email_id = str(uuid.uuid4())[:12]
        recipient = email_data.get("email", "")
        html_content = email_data.get("html", "<p>Email content</p>")
        subject = email_data.get("subject", "Your exclusive BFSI offer")

        # Inject tracking pixel
        pixel = generate_tracking_pixel(email_id, recipient, campaign_id)
        html_content = html_content.replace("</body>", f"{pixel}</body>")

        if self.sg_client:
            # Live SendGrid send
            try:
                mail = Mail(
                    from_email=Email(self.from_email),
                    to_emails=To(recipient),
                    subject=f"{subject} [{variant}]",
                    html_content=Content("text/html", html_content)
                )
                response = self.sg_client.client.mail.send.post(request_body=mail.get())
                return {
                    "id": email_id,
                    "email": recipient,
                    "status": "sent",
                    "variant": variant,
                    "sendgrid_status": response.status_code,
                }
            except Exception as e:
                return {"id": email_id, "email": recipient, "status": "failed", "error": str(e)}
        else:
            # Demo mode simulation
            time.sleep(0.01)  # simulate network latency
            return {
                "id": email_id,
                "email": recipient,
                "status": "sent_demo",
                "variant": variant,
                "campaign_id": campaign_id,
                "subject": subject,
            }

    async def send_batch_async(self, emails: List[Dict], campaign_id: str, schedule: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Async batch sender — handles 1k+ emails efficiently.
        Alternates A/B variants for each batch.
        """
        results = []
        total = len(emails)
        logger_scheduler.info(f"ForgeScheduler: Sending {total} emails in batches of {self.BATCH_SIZE}...")

        batches = [emails[i:i + self.BATCH_SIZE] for i in range(0, total, self.BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            variant = "A" if batch_idx % 2 == 0 else "B"  # A/B alternation
            batch_results = []

            for email_data in batch:
                # Queue to Celery for scalability
                celery_stub.delay(self._send_single, email_data, campaign_id, variant)
                # Also execute directly for demo
                result = self._send_single(email_data, campaign_id, variant)
                batch_results.append(result)

            results.extend(batch_results)
            sent_count = sum(1 for r in batch_results if "sent" in r.get("status", ""))
            logger_scheduler.info(f"  Batch {batch_idx + 1}/{len(batches)}: {sent_count}/{len(batch)} sent (Variant {variant})")
            await asyncio.sleep(self.BATCH_DELAY)

        success = sum(1 for r in results if "sent" in r.get("status", ""))
        logger_scheduler.info(f"ForgeScheduler: Complete — {success}/{total} emails sent.")
        return results

    def send(self, email_list: List[Dict], schedule: Optional[List[Dict]] = None) -> List[Dict]:
        """Synchronous wrapper for batch sending."""
        campaign_id = f"campaign_{int(time.time())}"
        return asyncio.run(self.send_batch_async(email_list, campaign_id, schedule))

    def __call__(self, input_data: Any) -> List[Dict]:
        """NexusHub slot callable."""
        if isinstance(input_data, dict):
            return self.send(
                email_list=input_data.get("emails", []),
                schedule=input_data.get("schedule")
            )
        return []


# ============================================================
# NexusForge — ForgeAnalyzer Agent
# ============================================================
"""
Analyzes email campaign metrics using Pandas.
Detects anomalies via z-score (e.g., bounce > 20%).
Generates Plotly graph stubs for dashboard.
Output: JSON metrics dict.
"""

import scipy.stats as stats

logger_analyzer = logging.getLogger("ForgeAnalyzer")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ForgeAnalyzer:
    """
    Analyzes tracking data for campaign performance.
    Flags anomalies using z-score detection.
    """

    ANOMALY_THRESHOLDS = {
        "bounce_rate": 20.0,    # >20% bounce is bad
        "unsubscribe_rate": 2.0, # >2% unsub is concerning
        "open_rate_low": 10.0,   # <10% open is poor
    }

    def analyze(self, tracking_data: List[Dict]) -> Dict:
        """
        Analyze campaign tracking data.

        Args:
            tracking_data: List of email event records.
            Each record: {email_id, email, status, opened, clicked, bounced, unsubscribed}

        Returns:
            JSON metrics dict with rates, anomalies, and plotly graph stubs.
        """
        try:
            import pandas as pd
            df = pd.DataFrame(tracking_data)
        except Exception as e:
            logger_analyzer.error(f"ForgeAnalyzer: Failed to load data: {e}")
            return {"error": str(e)}

        if df.empty:
            return {"error": "No tracking data provided"}

        total = len(df)

        # ── Core Metrics ──────────────────────────────────
        metrics = {
            "total_sent": total,
            "opened": int(df.get("opened", pd.Series([False]*total)).sum()),
            "clicked": int(df.get("clicked", pd.Series([False]*total)).sum()),
            "bounced": int(df.get("bounced", pd.Series([False]*total)).sum()),
            "unsubscribed": int(df.get("unsubscribed", pd.Series([False]*total)).sum()),
        }
        metrics["open_rate"] = round(metrics["opened"] / total * 100, 2) if total > 0 else 0
        metrics["click_rate"] = round(metrics["clicked"] / total * 100, 2) if total > 0 else 0
        metrics["bounce_rate"] = round(metrics["bounced"] / total * 100, 2) if total > 0 else 0
        metrics["ctr"] = round(metrics["clicked"] / max(metrics["opened"], 1) * 100, 2)
        metrics["unsubscribe_rate"] = round(metrics["unsubscribed"] / total * 100, 2)

        # ── A/B Variant Comparison ─────────────────────────
        if "variant" in df.columns:
            ab_metrics = {}
            for variant in df["variant"].unique():
                vdf = df[df["variant"] == variant]
                vtotal = len(vdf)
                ab_metrics[f"variant_{variant}"] = {
                    "count": vtotal,
                    "open_rate": round(vdf.get("opened", pd.Series([False]*vtotal)).sum() / vtotal * 100, 2),
                    "click_rate": round(vdf.get("clicked", pd.Series([False]*vtotal)).sum() / vtotal * 100, 2),
                }
            metrics["ab_comparison"] = ab_metrics

        # ── Anomaly Detection (Z-Score) ────────────────────
        anomalies = []

        # Check bounce rate anomaly
        if metrics["bounce_rate"] > self.ANOMALY_THRESHOLDS["bounce_rate"]:
            anomalies.append({
                "type": "high_bounce_rate",
                "value": metrics["bounce_rate"],
                "threshold": self.ANOMALY_THRESHOLDS["bounce_rate"],
                "severity": "HIGH" if metrics["bounce_rate"] > 30 else "MEDIUM",
                "action": "Review email list quality, check for invalid addresses",
            })

        # Check low open rate
        if metrics["open_rate"] < self.ANOMALY_THRESHOLDS["open_rate_low"]:
            anomalies.append({
                "type": "low_open_rate",
                "value": metrics["open_rate"],
                "threshold": self.ANOMALY_THRESHOLDS["open_rate_low"],
                "severity": "MEDIUM",
                "action": "A/B test subject lines, check send time, review spam score",
            })

        # Z-score detection on per-email data if numerical cols exist
        if "open_delay_seconds" in df.columns:
            try:
                z_scores = np.abs(stats.zscore(df["open_delay_seconds"].dropna()))
                outlier_count = int((z_scores > 3).sum())
                if outlier_count > 0:
                    anomalies.append({
                        "type": "timing_outliers",
                        "count": outlier_count,
                        "severity": "LOW",
                        "action": "Check for bot opens or delayed delivery",
                    })
            except Exception:
                pass

        metrics["anomalies"] = anomalies
        metrics["anomaly_count"] = len(anomalies)
        metrics["health_score"] = max(0, 100 - len(anomalies) * 15 - max(0, metrics["bounce_rate"] - 5))

        # ── Plotly Graph Stubs ──────────────────────────────
        metrics["plotly_stubs"] = self._generate_plotly_stubs(metrics)

        logger_analyzer.info(
            f"ForgeAnalyzer: Analyzed {total} emails. "
            f"Open={metrics['open_rate']}%, Click={metrics['click_rate']}%, "
            f"Bounce={metrics['bounce_rate']}%, Anomalies={len(anomalies)}"
        )
        return metrics

    def _generate_plotly_stubs(self, metrics: Dict) -> Dict:
        """Generate Plotly chart configs (data + layout) for dashboard rendering."""
        # Funnel chart: sent → opened → clicked
        funnel_stub = {
            "type": "funnel",
            "data": [{
                "type": "funnel",
                "y": ["Sent", "Opened", "Clicked"],
                "x": [metrics["total_sent"], metrics["opened"], metrics["clicked"]],
                "textinfo": "value+percent initial",
                "marker": {"color": ["#3b82f6", "#10b981", "#f59e0b"]},
            }],
            "layout": {"title": "Campaign Funnel", "height": 300},
        }

        # Metrics gauge chart
        gauge_stub = {
            "type": "indicator",
            "data": [{
                "type": "indicator",
                "mode": "gauge+number",
                "value": metrics["open_rate"],
                "title": {"text": "Open Rate %"},
                "gauge": {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 20], "color": "#fee2e2"},
                        {"range": [20, 40], "color": "#fef3c7"},
                        {"range": [40, 100], "color": "#d1fae5"},
                    ],
                },
            }],
            "layout": {"height": 250},
        }

        return {"funnel": funnel_stub, "open_rate_gauge": gauge_stub}

    def __call__(self, input_data: Any) -> Dict:
        """NexusHub slot callable."""
        if isinstance(input_data, dict):
            return self.analyze(input_data.get("tracking_data", []))
        if isinstance(input_data, list):
            return self.analyze(input_data)
        return {"error": "Invalid input"}


# ============================================================
# NexusForge — ForgeOptimizer Agent
# ============================================================
"""
RL-based optimizer using NumPy Q-Learning.
Runs A/B tests, updates weights, refines campaign plan.
Integrates foresight for parallel "what-if" simulations.
Max 3 optimization loops.
"""

logger_optimizer = logging.getLogger("ForgeOptimizer")


class ForgeOptimizer:
    """
    Q-Learning optimizer for campaign refinement.
    States: campaign performance level (low/mid/high)
    Actions: change_subject, change_time, change_segment, keep_same
    """

    ACTIONS = ["change_subject", "change_send_time", "narrow_segment", "increase_budget", "keep_same"]
    STATE_BINS = ["poor", "average", "good", "excellent"]
    MAX_LOOPS = 3

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.9, epsilon: float = 0.2):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon  # exploration rate

        # Q-table: states × actions
        n_states = len(self.STATE_BINS)
        n_actions = len(self.ACTIONS)
        self.q_table = np.random.uniform(0, 0.1, (n_states, n_actions))
        logger_optimizer.info("ForgeOptimizer initialized with NumPy Q-Learning.")

    def _get_state(self, metrics: Dict) -> int:
        """Map metrics to a state index."""
        open_rate = metrics.get("open_rate", 0)
        click_rate = metrics.get("click_rate", 0)
        score = (open_rate * 0.6) + (click_rate * 0.4)

        if score < 15:
            return 0  # poor
        elif score < 30:
            return 1  # average
        elif score < 50:
            return 2  # good
        else:
            return 3  # excellent

    def _get_reward(self, before: Dict, after: Dict) -> float:
        """Calculate reward based on metric improvement."""
        open_improvement = after.get("open_rate", 0) - before.get("open_rate", 0)
        click_improvement = after.get("click_rate", 0) - before.get("click_rate", 0)
        bounce_penalty = after.get("bounce_rate", 0) - before.get("bounce_rate", 0)
        return (open_improvement * 0.5) + (click_improvement * 0.5) - (bounce_penalty * 0.3)

    def _select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.ACTIONS))  # explore
        return int(np.argmax(self.q_table[state]))  # exploit

    def _update_q(self, state: int, action: int, reward: float, next_state: int):
        """Q-learning update rule."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def _run_foresight_simulations(self, plan: Dict, n_sims: int = 3) -> List[Dict]:
        """
        Run parallel 'what-if' simulations using Foresight Oracle.
        Explores different economic scenarios.
        """
        scenarios = [
            {"name": "baseline", "econ_multiplier": 1.0, "inflation": "normal"},
            {"name": "high_inflation", "econ_multiplier": 0.85, "inflation": "high"},
            {"name": "growth_phase", "econ_multiplier": 1.2, "inflation": "low"},
        ]

        sim_results = []
        for scenario in scenarios[:n_sims]:
            simulated_open_rate = plan.get("predicted_engagement", 50) * scenario["econ_multiplier"]
            sim_results.append({
                "scenario": scenario["name"],
                "simulated_open_rate": round(simulated_open_rate, 2),
                "recommendation": (
                    "Delay campaign" if scenario["econ_multiplier"] < 0.9
                    else "Proceed aggressively" if scenario["econ_multiplier"] > 1.1
                    else "Standard execution"
                ),
            })

        return sim_results

    def optimize(self, metrics: Dict, current_plan: Dict) -> Dict:
        """
        Run optimization loop (max 3 iterations) to refine the campaign plan.

        Args:
            metrics: Current campaign analytics from ForgeAnalyzer.
            current_plan: Current plan dict from ForgePlanner.

        Returns:
            Refined plan with optimization actions and Q-table state.
        """
        logger_optimizer.info("ForgeOptimizer: Starting optimization...")

        refined_plan = dict(current_plan)
        optimization_log = []
        current_metrics = dict(metrics)

        for loop in range(self.MAX_LOOPS):
            state = self._get_state(current_metrics)
            action_idx = self._select_action(state)
            action = self.ACTIONS[action_idx]

            logger_optimizer.info(f"  Loop {loop + 1}: State={self.STATE_BINS[state]}, Action={action}")

            # Apply action to plan
            next_metrics = dict(current_metrics)
            action_effect = {}

            if action == "change_subject":
                refined_plan["outline"] = refined_plan.get("outline", {})
                refined_plan["outline"]["subject_line"] = "🚨 LAST CHANCE: " + refined_plan.get("outline", {}).get("subject_line", "Special Offer")
                next_metrics["open_rate"] = current_metrics.get("open_rate", 0) * 1.15
                action_effect = {"description": "Made subject line more urgent", "expected_open_lift": "+15%"}

            elif action == "change_send_time":
                for item in refined_plan.get("schedule", []):
                    item["send_time"] = "10:00 AM IST"  # optimize to best time
                next_metrics["open_rate"] = current_metrics.get("open_rate", 0) * 1.08
                action_effect = {"description": "Shifted to optimal send time (10 AM IST)", "expected_open_lift": "+8%"}

            elif action == "narrow_segment":
                segments = refined_plan.get("segments", [])
                if len(segments) > 1:
                    # Keep only highest engagement segment
                    refined_plan["segments"] = [max(segments, key=lambda s: s.get("avg_past_opens", 0))]
                next_metrics["click_rate"] = current_metrics.get("click_rate", 0) * 1.2
                action_effect = {"description": "Narrowed to highest-engagement segment", "expected_ctr_lift": "+20%"}

            elif action == "increase_budget":
                next_metrics["open_rate"] = current_metrics.get("open_rate", 0) * 1.1
                action_effect = {"description": "Suggested budget increase for paid reach", "expected_open_lift": "+10%"}

            else:  # keep_same
                action_effect = {"description": "Maintaining current plan — already optimal", "change": "none"}

            # Calculate reward and update Q-table
            reward = self._get_reward(current_metrics, next_metrics)
            next_state = self._get_state(next_metrics)
            self._update_q(state, action_idx, reward, next_state)

            optimization_log.append({
                "loop": loop + 1,
                "state": self.STATE_BINS[state],
                "action": action,
                "reward": round(reward, 3),
                "action_effect": action_effect,
            })

            current_metrics = next_metrics

        # Run foresight simulations
        foresight_sims = self._run_foresight_simulations(refined_plan)

        refined_plan["optimization"] = {
            "loops_run": self.MAX_LOOPS,
            "optimization_log": optimization_log,
            "final_metrics_estimate": current_metrics,
            "q_table_sample": self.q_table.tolist(),
            "foresight_simulations": foresight_sims,
            "recommendation": optimization_log[-1]["action_effect"].get("description", "No change"),
        }

        logger_optimizer.info(f"ForgeOptimizer: Complete. Final est. open_rate={current_metrics.get('open_rate', 0):.1f}%")
        return refined_plan

    def __call__(self, input_data: Any) -> Dict:
        """NexusHub slot callable."""
        if isinstance(input_data, dict):
            return self.optimize(
                metrics=input_data.get("metrics", {}),
                current_plan=input_data.get("plan", {})
            )
        return {}
