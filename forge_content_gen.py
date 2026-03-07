"""
============================================================
NexusForge — ForgeContentGen Agent
============================================================
Generates personalized HTML emails using Jinja2 templates.
Creates dynamic PNG charts (Matplotlib) for each customer.
Applies sentiment refinement for hyper-personalization.
Outputs list of HTML strings (5 BFSI template variants).
============================================================
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from jinja2 import Environment, BaseLoader

# ── VADER Sentiment (graceful import) ──────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger("ForgeContentGen")


# ══════════════════════════════════════════════════════════════
# Jinja2 HTML Templates — 5 BFSI Variants
# ══════════════════════════════════════════════════════════════

TEMPLATES = {
    "loan_offer": """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body { font-family: Arial, sans-serif; background: #f5f7fa; margin: 0; padding: 0; }
  .container { max-width: 600px; margin: auto; background: white; border-radius: 12px; overflow: hidden; }
  .header { background: linear-gradient(135deg, #1a3a6b, #2563eb); color: white; padding: 32px; text-align: center; }
  .header h1 { margin: 0; font-size: 26px; }
  .badge { background: #f59e0b; color: #111; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 10px; display: inline-block; }
  .body { padding: 28px; }
  .greeting { font-size: 18px; color: #1e293b; margin-bottom: 16px; }
  .loan-box { background: #eff6ff; border-left: 4px solid #2563eb; padding: 16px; border-radius: 8px; margin: 20px 0; }
  .loan-box h3 { margin: 0 0 8px 0; color: #1e40af; }
  .loan-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #dbeafe; }
  .cta { background: #2563eb; color: white; padding: 14px 32px; text-decoration: none; border-radius: 8px; display: inline-block; font-weight: bold; margin: 20px 0; }
  .chart-section { text-align: center; padding: 12px; }
  .footer { background: #f1f5f9; padding: 16px; text-align: center; font-size: 11px; color: #94a3b8; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="badge">PRE-APPROVED ✓</div>
    <h1>Exclusive Home Loan Offer</h1>
    <p>Specially crafted for you, {{ customer.name }}</p>
  </div>
  <div class="body">
    <div class="greeting">Dear {{ customer.name }},</div>
    <p>Based on your {{ income_band }} profile in <strong>{{ customer.location }}</strong>, you qualify for our exclusive home loan at industry-low rates.</p>
    <div class="loan-box">
      <h3>🏠 Your Loan Details</h3>
      <div class="loan-row"><span>Loan Amount</span><span><strong>Up to ₹{{ loan_amount }}</strong></span></div>
      <div class="loan-row"><span>Interest Rate</span><span><strong>{{ interest_rate }}% p.a.</strong></span></div>
      <div class="loan-row"><span>Tenure</span><span><strong>Up to 30 years</strong></span></div>
      <div class="loan-row"><span>Processing Fee</span><span><strong>₹0 (Waived!)</strong></span></div>
    </div>
    <div class="chart-section">
      <img src="data:image/png;base64,{{ chart_b64 }}" alt="EMI Chart" style="width:100%;border-radius:8px;">
    </div>
    <p style="color:#64748b;font-size:13px;">{{ sentiment_message }}</p>
    <center><a href="#" class="cta">Apply Now in 2 Minutes →</a></center>
  </div>
  <div class="footer">
    NexusForge BFSI Platform | {{ customer.email }} | Unsubscribe<br>
    Offer valid for 7 days. Subject to credit approval. T&C apply.
  </div>
</div>
</body></html>""",

    "investment": """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body { font-family: Georgia, serif; background: #0f172a; margin: 0; }
  .container { max-width: 600px; margin: auto; background: #1e293b; color: white; border-radius: 12px; overflow: hidden; }
  .header { background: linear-gradient(135deg, #064e3b, #059669); padding: 32px; text-align: center; }
  .header h1 { color: #d1fae5; margin: 0; }
  .body { padding: 28px; }
  .fund-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin: 10px 0; }
  .fund-row { display: flex; justify-content: space-between; padding: 4px 0; color: #94a3b8; }
  .returns { color: #34d399; font-size: 22px; font-weight: bold; }
  .cta { background: #059669; color: white; padding: 14px 32px; border-radius: 8px; display: inline-block; font-weight: bold; text-decoration: none; }
  .chart-section { text-align: center; margin: 16px 0; }
  .footer { background: #0f172a; padding: 14px; text-align: center; font-size: 11px; color: #475569; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📈 Grow Your Wealth</h1>
    <p style="color:#a7f3d0;">Curated Investment Plans for {{ customer.name }}</p>
  </div>
  <div class="body">
    <p>Hello {{ customer.name }},<br>Based on your profile, here are top-performing funds for {{ income_band }} investors in {{ customer.location }}:</p>
    <div class="fund-card">
      <div class="fund-row"><span>Bluechip Equity Fund</span><span class="returns">+{{ returns_1y }}% (1Y)</span></div>
      <div class="fund-row"><span>Risk</span><span>Moderate</span></div>
    </div>
    <div class="fund-card">
      <div class="fund-row"><span>Balanced Advantage Fund</span><span class="returns">+{{ returns_3y }}% (3Y)</span></div>
      <div class="fund-row"><span>Risk</span><span>Low-Moderate</span></div>
    </div>
    <div class="chart-section">
      <img src="data:image/png;base64,{{ chart_b64 }}" alt="Returns Chart" style="width:100%;border-radius:8px;">
    </div>
    <p style="color:#94a3b8;font-size:13px;">{{ sentiment_message }}</p>
    <center><a href="#" class="cta">Start SIP Today →</a></center>
  </div>
  <div class="footer">NexusForge | Mutual fund investments are subject to market risks.</div>
</div></body></html>""",

    "credit_card": """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #fafafa; }
  .container { max-width: 600px; margin: auto; background: white; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.1); overflow: hidden; }
  .header { background: linear-gradient(135deg, #7c3aed, #4f46e5); padding: 32px; text-align: center; }
  .header h1 { color: white; margin: 0; }
  .body { padding: 28px; }
  .card-visual { background: linear-gradient(135deg, #6d28d9, #4338ca); color: white; border-radius: 16px; padding: 20px; margin: 16px 0; font-family: monospace; }
  .benefit-item { display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #f1f5f9; }
  .benefit-icon { font-size: 20px; margin-right: 12px; }
  .cta { background: #7c3aed; color: white; padding: 14px 32px; border-radius: 8px; display: inline-block; font-weight: bold; text-decoration: none; }
  .footer { background: #f8fafc; padding: 14px; text-align: center; font-size: 11px; color: #94a3b8; }
</style>
</head>
<body>
<div class="container">
  <div class="header"><h1>💎 Your Premium Upgrade</h1></div>
  <div class="body">
    <div class="card-visual">
      <div style="font-size:18px;font-weight:bold;">NexusForge Platinum</div>
      <div style="margin-top:20px;font-size:22px;letter-spacing:4px;">•••• •••• •••• 4242</div>
      <div style="margin-top:10px;">{{ customer.name.upper() }}</div>
    </div>
    <p>Dear {{ customer.name }}, your {{ income_band }} profile qualifies you for our Platinum Card:</p>
    <div class="benefit-item"><span class="benefit-icon">✈️</span><span>5X rewards on travel & dining</span></div>
    <div class="benefit-item"><span class="benefit-icon">🛡️</span><span>₹50L complimentary insurance</span></div>
    <div class="benefit-item"><span class="benefit-icon">💰</span><span>2% cashback on all spends</span></div>
    <div class="benefit-item"><span class="benefit-icon">🏨</span><span>Priority Pass lounge access</span></div>
    <p style="color:#64748b;font-size:13px;">{{ sentiment_message }}</p>
    <center><a href="#" class="cta">Apply for Platinum Card →</a></center>
  </div>
  <div class="footer">NexusForge BFSI | Credit subject to bank approval.</div>
</div></body></html>""",

    "insurance": """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body { font-family: Arial, sans-serif; background: #f0fdf4; }
  .container { max-width: 600px; margin: auto; background: white; border-radius: 12px; overflow: hidden; }
  .header { background: linear-gradient(135deg, #065f46, #10b981); padding: 28px; text-align: center; color: white; }
  .body { padding: 28px; }
  .coverage-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  .coverage-table th { background: #ecfdf5; padding: 10px; text-align: left; color: #065f46; }
  .coverage-table td { padding: 10px; border-bottom: 1px solid #d1fae5; }
  .price-badge { background: #10b981; color: white; padding: 6px 16px; border-radius: 20px; font-size: 18px; font-weight: bold; }
  .cta { background: #10b981; color: white; padding: 14px 32px; border-radius: 8px; display: inline-block; font-weight: bold; text-decoration: none; }
  .footer { background: #ecfdf5; padding: 14px; text-align: center; font-size: 11px; color: #6b7280; }
</style>
</head>
<body>
<div class="container">
  <div class="header"><h1>🛡️ Protect What Matters</h1><p>Tailored for {{ customer.name }} in {{ customer.location }}</p></div>
  <div class="body">
    <p>Dear {{ customer.name }}, at just <span class="price-badge">₹99/month</span> secure your family's future:</p>
    <table class="coverage-table">
      <tr><th>Coverage Type</th><th>Amount</th><th>Premium</th></tr>
      <tr><td>Life Cover</td><td>₹1 Crore</td><td>₹599/mo</td></tr>
      <tr><td>Health Cover</td><td>₹10 Lakh</td><td>₹299/mo</td></tr>
      <tr><td>Accidental Cover</td><td>₹50 Lakh</td><td>₹99/mo</td></tr>
    </table>
    <div style="text-align:center;"><img src="data:image/png;base64,{{ chart_b64 }}" style="width:90%;border-radius:8px;"></div>
    <p style="color:#6b7280;font-size:13px;">{{ sentiment_message }}</p>
    <center><a href="#" class="cta">Get Free Quote →</a></center>
  </div>
  <div class="footer">NexusForge Insurance | IRDAI Registered | T&C apply.</div>
</div></body></html>""",

    "fd_offer": """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body { font-family: Arial, sans-serif; background: #fffbeb; }
  .container { max-width: 600px; margin: auto; background: white; border-radius: 12px; overflow: hidden; }
  .header { background: linear-gradient(135deg, #92400e, #f59e0b); padding: 28px; text-align: center; color: white; }
  .rate-hero { font-size: 52px; font-weight: bold; color: #b45309; text-align: center; margin: 20px 0; }
  .rate-sub { text-align: center; color: #78350f; font-size: 16px; margin-top: -10px; }
  .body { padding: 28px; }
  .table { width: 100%; border-collapse: collapse; }
  .table th { background: #fef3c7; padding: 10px; color: #78350f; }
  .table td { padding: 10px; border-bottom: 1px solid #fde68a; }
  .cta { background: #f59e0b; color: #111; padding: 14px 32px; border-radius: 8px; display: inline-block; font-weight: bold; text-decoration: none; }
  .footer { background: #fffbeb; padding: 14px; text-align: center; font-size: 11px; color: #a16207; }
</style>
</head>
<body>
<div class="container">
  <div class="header"><h1>🏦 Fixed Deposit Special Rate</h1></div>
  <div class="rate-hero">8.5%</div>
  <div class="rate-sub">p.a. — For {{ income_band }} customers in {{ customer.location }}</div>
  <div class="body">
    <p>Dear {{ customer.name }}, lock in guaranteed returns before rates change:</p>
    <table class="table">
      <tr><th>Tenure</th><th>Rate (p.a.)</th><th>₹1L grows to</th></tr>
      <tr><td>1 Year</td><td>7.5%</td><td>₹1,07,500</td></tr>
      <tr><td>2 Years</td><td>8.0%</td><td>₹1,16,640</td></tr>
      <tr><td>3 Years</td><td>8.5%</td><td>₹1,27,788</td></tr>
    </table>
    <div style="text-align:center;margin:16px 0;"><img src="data:image/png;base64,{{ chart_b64 }}" style="width:90%;border-radius:8px;"></div>
    <p style="color:#78350f;font-size:13px;">{{ sentiment_message }}</p>
    <center><a href="#" class="cta">Book FD Now — Limited Period →</a></center>
  </div>
  <div class="footer">NexusForge | DICGC Insured up to ₹5 Lakh | RBI Guidelines apply.</div>
</div></body></html>""",
}


# ══════════════════════════════════════════════════════════════
# Dynamic Chart Generator (Matplotlib → Base64 PNG)
# ══════════════════════════════════════════════════════════════
def generate_dynamic_chart(template_type: str, customer: Dict) -> str:
    """
    Generate a personalized PNG chart based on template type.
    Returns base64-encoded PNG string for embedding in HTML.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    income = customer.get("income", 500000)

    if template_type == "loan_offer":
        # EMI comparison chart across tenures
        tenures = [10, 15, 20, 25, 30]
        loan_amount = min(income * 5, 5000000)
        rate = 8.5 / (12 * 100)
        emis = [loan_amount * rate * (1 + rate)**n / ((1 + rate)**n - 1) / 1000 for n in [y*12 for y in tenures]]
        bars = ax.bar(tenures, emis, color=["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"], width=3)
        ax.set_xlabel("Tenure (years)", fontsize=9)
        ax.set_ylabel("EMI (₹ thousands)", fontsize=9)
        ax.set_title(f"Your EMI Across Tenures — Loan ₹{loan_amount/100000:.0f}L @ 8.5%", fontsize=10, fontweight="bold")
        for bar, emi in zip(bars, emis):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"₹{emi:.0f}k", ha='center', va='bottom', fontsize=8)

    elif template_type == "investment":
        # Wealth growth projection
        years = list(range(0, 11))
        sip_monthly = max(5000, income * 0.05 / 12)
        # 12% annual return
        wealth = [sip_monthly * 12 * y * (1.12 ** (y/2)) for y in years]
        ax.fill_between(years, wealth, alpha=0.3, color="#10b981")
        ax.plot(years, wealth, color="#059669", linewidth=2.5, marker="o", markersize=4)
        ax.set_xlabel("Years", fontsize=9)
        ax.set_ylabel("Wealth (₹)", fontsize=9)
        ax.set_title(f"SIP Growth — ₹{sip_monthly:.0f}/mo at 12% p.a.", fontsize=10, fontweight="bold")

    elif template_type == "fd_offer":
        # FD maturity value comparison
        tenures = ["1 Year\n7.5%", "2 Years\n8.0%", "3 Years\n8.5%"]
        rates = [0.075, 0.08, 0.085]
        principal = max(50000, income * 0.1)
        values = [principal * (1 + r)**i for i, r in zip([1, 2, 3], rates)]
        bars = ax.bar(tenures, values, color=["#fbbf24", "#f59e0b", "#d97706"], width=0.5)
        ax.axhline(y=principal, color="gray", linestyle="--", linewidth=1, label=f"Principal ₹{principal:,.0f}")
        ax.set_ylabel("Maturity Value (₹)", fontsize=9)
        ax.set_title("FD Maturity Value Comparison", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + principal*0.01,
                    f"₹{val:,.0f}", ha='center', va='bottom', fontsize=8)

    elif template_type == "insurance":
        # Coverage vs premium pie chart
        labels = ["Life Cover", "Health Cover", "Accident Cover"]
        sizes = [60, 30, 10]
        colors = ["#10b981", "#34d399", "#6ee7b7"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 9})
        ax.set_title("Coverage Distribution", fontsize=10, fontweight="bold")

    else:
        # Generic bar chart
        categories = ["Open Rate", "Click Rate", "Conversion", "ROI"]
        values = [45, 28, 12, 320]
        ax.bar(categories, values, color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"])
        ax.set_title("Campaign Performance Benchmark", fontsize=10, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
# Sentiment Analyzer
# ══════════════════════════════════════════════════════════════
def get_sentiment_message(customer: Dict, template_type: str) -> str:
    """
    Generate a personalized sentiment-matched message.
    Uses VADER if available, else rule-based matching.
    """
    past_opens = customer.get("past_opens", 0)
    name = customer.get("name", "Valued Customer")

    # Infer customer mood from past engagement
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        # Simulate a past interaction text (in production, use actual CRM notes)
        simulated_text = f"Customer {name} has opened {past_opens} emails recently."
        score = analyzer.polarity_scores(simulated_text)["compound"]
    else:
        score = (past_opens - 3) / 5  # normalize past opens to [-1, 1]

    # Match message to sentiment
    if score > 0.3:  # positive / engaged
        messages = {
            "loan_offer": f"Great news, {name}! Your strong financial history makes you our priority customer. 🌟",
            "investment": f"You're already ahead of the curve, {name}. Let's grow your wealth further! 💹",
            "credit_card": f"Your excellent engagement earns you our best card, {name}! 🏆",
            "insurance": f"Smart move, {name}! Securing your future is the wisest investment. 🛡️",
            "fd_offer": f"You know the value of saving, {name}. Lock in the best rates today! 🔒",
        }
    elif score < -0.3:  # negative / disengaged
        messages = {
            "loan_offer": f"We understand you're busy, {name}. This 2-minute application could save you lakhs. ⏱️",
            "investment": f"Even small steps matter, {name}. Start with just ₹500/month. 🌱",
            "credit_card": f"No pressure, {name}. Explore at your own pace — no annual fee first year. 😊",
            "insurance": f"We know life is hectic, {name}. One quick call, and your family is protected. 🤝",
            "fd_offer": f"Safe and simple, {name}. No market risk, guaranteed returns. 🔐",
        }
    else:  # neutral
        messages = {
            "loan_offer": f"Tailored exclusively for you, {name}. Your dream home awaits. 🏠",
            "investment": f"The right time to invest is now, {name}. Beat inflation with smart returns. 📊",
            "credit_card": f"Premium benefits, zero stress, {name}. Our card, your lifestyle. ✨",
            "insurance": f"Peace of mind for you and your family, {name}. That's priceless. 💙",
            "fd_offer": f"Guaranteed returns await you, {name}. Book before rates drop. ⏰",
        }

    return messages.get(template_type, f"Special offer for you, {name}!")


# ══════════════════════════════════════════════════════════════
# ForgeContentGen Agent — Main Class
# ══════════════════════════════════════════════════════════════
class ForgeContentGen:
    """
    Generates personalized HTML email content for BFSI campaigns.

    Features:
    - 5 Jinja2 HTML templates (loan, investment, credit card, insurance, FD)
    - Dynamic Matplotlib charts embedded as base64 PNG
    - VADER sentiment analysis for hyper-personalization
    - Per-customer content variation
    """

    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader())
        logger.info("ForgeContentGen initialized with 5 BFSI templates + dynamic chart engine.")

    def _personalize_vars(self, customer: Dict, template_type: str) -> Dict:
        """Build template variables for a specific customer."""
        income = customer.get("income", 300000)

        # Income band
        if income >= 800000:
            income_band = "Premium"
            loan_amount = "75 Lakh"
            interest_rate = "8.25"
            returns_1y = "22.4"
            returns_3y = "68.9"
        elif income >= 300000:
            income_band = "Standard"
            loan_amount = "40 Lakh"
            interest_rate = "8.75"
            returns_1y = "18.2"
            returns_3y = "52.1"
        else:
            income_band = "Value"
            loan_amount = "20 Lakh"
            interest_rate = "9.25"
            returns_1y = "15.6"
            returns_3y = "44.7"

        return {
            "customer": customer,
            "income_band": income_band,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "returns_1y": returns_1y,
            "returns_3y": returns_3y,
        }

    def generate(self, plan: Dict, customers: List[Dict]) -> List[Dict]:
        """
        Generate personalized HTML emails for each customer.

        Args:
            plan: Campaign plan from ForgePlanner.
            customers: List of customer dicts.

        Returns:
            List of dicts: {email, html, template_type, customer_name}
        """
        template_type = plan.get("outline", {}).get("template_type", "loan_offer")
        results = []

        logger.info(f"ForgeContentGen: Generating emails for {len(customers)} customers, template={template_type}")

        for customer in customers:
            try:
                # Build template variables
                vars_dict = self._personalize_vars(customer, template_type)

                # Generate dynamic chart
                chart_b64 = generate_dynamic_chart(template_type, customer)
                vars_dict["chart_b64"] = chart_b64

                # Sentiment message
                vars_dict["sentiment_message"] = get_sentiment_message(customer, template_type)

                # Render HTML
                template_str = TEMPLATES.get(template_type, TEMPLATES["loan_offer"])
                rendered_html = self.jinja_env.from_string(template_str).render(**vars_dict)

                results.append({
                    "email": customer.get("email", ""),
                    "customer_name": customer.get("name", ""),
                    "template_type": template_type,
                    "html": rendered_html,
                    "personalized": True,
                })

            except Exception as e:
                logger.error(f"ForgeContentGen: Failed for {customer.get('email', '?')}: {e}")
                results.append({
                    "email": customer.get("email", ""),
                    "customer_name": customer.get("name", ""),
                    "error": str(e),
                    "html": None,
                })

        logger.info(f"ForgeContentGen: Generated {len([r for r in results if r.get('html')])} HTML emails.")
        return results

    def __call__(self, input_data: Any) -> List[Dict]:
        """Make ForgeContentGen callable for NexusHub slot registration."""
        if isinstance(input_data, dict):
            return self.generate(
                plan=input_data.get("plan", {}),
                customers=input_data.get("customers", [])
            )
        return []
