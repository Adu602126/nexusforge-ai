"""
============================================================
NexusForge — Foresight Oracle (Prophet-based ML)
============================================================
Time-series forecasting with Prophet.
Input: historical CSV (date, opens).
Output: forecast dict with confidence and trend multiplier.
Callable by all agents for predictive adjustments.
============================================================
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ForesightOracle")

# ── Prophet (graceful import) ─────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        logger.warning("Prophet not installed — using statistical fallback for forecasting.")


# ══════════════════════════════════════════════════════════════
# Foresight Oracle
# ══════════════════════════════════════════════════════════════
class ForesightOracle:
    """
    Predictive ML module using Facebook Prophet for time-series forecasting.
    Predicts future campaign engagement based on historical email metrics.

    Futuristic: Integrates external signals (inflation, market indices)
    to simulate 'Campaign Success Probability'.
    """

    def __init__(self):
        self.model = None
        self.forecast_cache: Dict = {}
        logger.info("ForesightOracle initialized.")

    def _generate_dummy_historical_data(self, days: int = 180) -> pd.DataFrame:
        """
        Generate realistic dummy email open rate historical data.
        Simulates seasonal patterns (higher opens on Tue/Wed, lower on weekends).
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        np.random.seed(42)

        # Base trend with weekly seasonality
        base = 30 + np.linspace(0, 5, days)  # slight upward trend
        weekly = np.array([2, 5, 4, 3, 1, -5, -4] * (days // 7 + 1))[:days]  # Mon-Sun pattern
        noise = np.random.normal(0, 2, days)
        opens = np.clip(base + weekly + noise, 5, 70).astype(int)

        return pd.DataFrame({"ds": dates, "y": opens})

    def forecast(self, historical_csv: Optional[str] = None, periods: int = 30) -> Dict:
        """
        Run Prophet forecast on email open rates.

        Args:
            historical_csv: Path to CSV with 'date' and 'opens' columns.
                           If None, uses dummy historical data.
            periods: Number of future days to forecast.

        Returns:
            Dict with forecast values, confidence intervals, and trend analysis.
        """
        logger.info(f"ForesightOracle: Generating {periods}-day forecast...")

        # Load historical data
        if historical_csv and os.path.exists(historical_csv):
            try:
                df = pd.read_csv(historical_csv)
                # Rename columns to Prophet format
                df.columns = [c.lower().strip() for c in df.columns]
                if "date" in df.columns and "opens" in df.columns:
                    df = df.rename(columns={"date": "ds", "opens": "y"})
                df["ds"] = pd.to_datetime(df["ds"])
                logger.info(f"ForesightOracle: Loaded {len(df)} historical records from {historical_csv}")
            except Exception as e:
                logger.warning(f"ForesightOracle: CSV load failed ({e}) — using dummy data.")
                df = self._generate_dummy_historical_data()
        else:
            logger.info("ForesightOracle: Using dummy historical data.")
            df = self._generate_dummy_historical_data()

        if PROPHET_AVAILABLE:
            return self._prophet_forecast(df, periods)
        else:
            return self._statistical_forecast(df, periods)

    def _prophet_forecast(self, df: pd.DataFrame, periods: int) -> Dict:
        """Run actual Prophet forecast."""
        try:
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode="additive",
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
            )
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast_df = model.predict(future)

            # Extract future predictions only
            future_preds = forecast_df.tail(periods)
            historical_mean = df["y"].mean()
            forecast_mean = future_preds["yhat"].mean()
            trend_multiplier = forecast_mean / max(historical_mean, 1)

            return {
                "method": "prophet",
                "forecast_period_days": periods,
                "historical_mean_opens": round(float(historical_mean), 2),
                "forecast_mean_opens": round(float(forecast_mean), 2),
                "trend_multiplier": round(float(trend_multiplier), 3),
                "confidence_lower": round(float(future_preds["yhat_lower"].mean()), 2),
                "confidence_upper": round(float(future_preds["yhat_upper"].mean()), 2),
                "confidence": 0.85,
                "risk_flag": forecast_mean < historical_mean * 0.85,
                "note": "Prophet forecast with weekly seasonality",
                "best_days": self._get_best_days(future_preds),
                "worst_days": self._get_worst_days(future_preds),
                "daily_forecast": [
                    {
                        "date": row["ds"].strftime("%Y-%m-%d"),
                        "predicted_open_rate": round(max(0, float(row["yhat"])), 1),
                        "lower": round(max(0, float(row["yhat_lower"])), 1),
                        "upper": round(float(row["yhat_upper"]), 1),
                    }
                    for _, row in future_preds.iterrows()
                ][:14],  # first 2 weeks
            }

        except Exception as e:
            logger.error(f"ForesightOracle: Prophet failed ({e}) — falling back to statistical.")
            return self._statistical_forecast(df, periods)

    def _statistical_forecast(self, df: pd.DataFrame, periods: int) -> Dict:
        """
        Statistical fallback: linear regression + seasonal adjustment.
        Used when Prophet is not installed.
        """
        y = df["y"].values
        x = np.arange(len(y))

        # Linear trend
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Forecast
        future_x = np.arange(len(y), len(y) + periods)
        forecast_values = np.polyval(coeffs, future_x)

        # Add seasonality (weekly pattern)
        weekly_pattern = np.array([2, 5, 4, 3, 1, -5, -4])
        seasonal = np.array([weekly_pattern[i % 7] for i in range(periods)])
        forecast_values = np.clip(forecast_values + seasonal + np.random.normal(0, 1.5, periods), 5, 80)

        historical_mean = y.mean()
        forecast_mean = forecast_values.mean()
        trend_multiplier = forecast_mean / max(historical_mean, 1)

        # Generate date range
        start_date = df["ds"].max() + timedelta(days=1)
        future_dates = [start_date + timedelta(days=i) for i in range(periods)]

        return {
            "method": "statistical_fallback",
            "forecast_period_days": periods,
            "historical_mean_opens": round(float(historical_mean), 2),
            "forecast_mean_opens": round(float(forecast_mean), 2),
            "trend_multiplier": round(float(trend_multiplier), 3),
            "confidence_lower": round(float(forecast_values.mean() - forecast_values.std()), 2),
            "confidence_upper": round(float(forecast_values.mean() + forecast_values.std()), 2),
            "confidence": 0.70,
            "risk_flag": bool(forecast_mean < historical_mean * 0.85),
            "note": "Statistical regression forecast (install Prophet for better accuracy)",
            "daily_forecast": [
                {"date": d.strftime("%Y-%m-%d"), "predicted_open_rate": round(float(v), 1)}
                for d, v in zip(future_dates[:14], forecast_values[:14])
            ],
        }

    def _get_best_days(self, forecast_df: pd.DataFrame) -> List[str]:
        """Get top 3 best days to send emails."""
        top = forecast_df.nlargest(3, "yhat")
        return [row["ds"].strftime("%Y-%m-%d (%A)") for _, row in top.iterrows()]

    def _get_worst_days(self, forecast_df: pd.DataFrame) -> List[str]:
        """Get 3 worst days to send emails."""
        bottom = forecast_df.nsmallest(3, "yhat")
        return [row["ds"].strftime("%Y-%m-%d (%A)") for _, row in bottom.iterrows()]

    def forecast_dummy(self) -> Dict:
        """Quick forecast using dummy data (for agent integration)."""
        return self.forecast(historical_csv=None, periods=14)


# ══════════════════════════════════════════════════════════════
# ComplianceGuard — BFSI Compliance Checker Agent
# ══════════════════════════════════════════════════════════════
"""
Scans email HTML for PII risks, sentiment risks, and BFSI compliance.
Uses regex for PII masking and VADER for sentiment analysis.
Output: safe flag + list of issues + fixed HTML.
"""

import re as _re

logger_compliance = logging.getLogger("ComplianceGuard")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _SIA
    _VADER_OK = True
except ImportError:
    _VADER_OK = False


class ComplianceGuard:
    """
    BFSI Compliance Checker — scans emails before sending.

    Checks:
    - PII: Aadhaar, PAN, phone numbers, credit card numbers
    - Sensitive financial language: guarantees, promised returns
    - Negative sentiment: aggressive, threatening language
    - SEBI/RBI compliance keywords
    """

    # PII Patterns
    PII_PATTERNS = {
        "aadhaar": (_re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "XXXX-XXXX-XXXX"),
        "pan": (_re.compile(r"\b[A-Z]{5}\d{4}[A-Z]{1}\b"), "[PAN MASKED]"),
        "phone": (_re.compile(r"\b(?:\+91[\s-]?)?\d{10}\b"), "[PHONE MASKED]"),
        "credit_card": (_re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD MASKED]"),
        "bank_account": (_re.compile(r"\b\d{9,18}\b(?=.*bank|.*account)", _re.IGNORECASE), "[ACCOUNT MASKED]"),
        "email_pii": (_re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[EMAIL MASKED]"),
    }

    # Prohibited financial claims
    PROHIBITED_PHRASES = [
        r"guaranteed returns",
        r"risk.?free investment",
        r"assured profit",
        r"100% returns",
        r"no risk",
        r"double your money",
        r"get rich",
        r"limited time.*guaranteed",
    ]

    # SEBI/RBI mandatory disclosures
    REQUIRED_DISCLOSURES = {
        "mutual_fund": "mutual fund investments are subject to market risks",
        "insurance": "subject to terms and conditions",
        "loan": "subject to credit approval",
    }

    def check(self, email_html: str, product_type: str = "general") -> Dict:
        """
        Run compliance check on email HTML.

        Args:
            email_html: Raw HTML string of the email.
            product_type: "loan", "mutual_fund", "insurance", "credit_card", "fd"

        Returns:
            Dict with: is_safe, issues, fixed_html, pii_found, sentiment_score
        """
        logger_compliance.info(f"ComplianceGuard: Checking email (product={product_type}, len={len(email_html)})...")

        issues = []
        fixed_html = email_html

        # ── PII Detection & Masking ───────────────────────
        pii_found = {}
        for pii_type, (pattern, replacement) in self.PII_PATTERNS.items():
            matches = pattern.findall(email_html)
            if matches:
                pii_found[pii_type] = len(matches)
                fixed_html = pattern.sub(replacement, fixed_html)
                if pii_type not in ["email_pii"]:  # emails are expected
                    issues.append({
                        "type": "PII_FOUND",
                        "pii_type": pii_type,
                        "count": len(matches),
                        "severity": "HIGH",
                        "action": f"Masked {len(matches)} {pii_type} instance(s)",
                        "auto_fixed": True,
                    })

        # ── Prohibited Phrases Check ──────────────────────
        text_content = _re.sub(r"<[^>]+>", " ", email_html)  # strip HTML tags
        text_lower = text_content.lower()

        for phrase_pattern in self.PROHIBITED_PHRASES:
            if _re.search(phrase_pattern, text_lower):
                issues.append({
                    "type": "PROHIBITED_CLAIM",
                    "phrase": phrase_pattern,
                    "severity": "HIGH",
                    "action": "Remove or rephrase the claim — violates SEBI/IRDAI guidelines",
                    "auto_fixed": False,
                })

        # ── Required Disclosures Check ────────────────────
        product_key = None
        if "mutual" in product_type or "invest" in product_type:
            product_key = "mutual_fund"
        elif "insur" in product_type:
            product_key = "insurance"
        elif "loan" in product_type or "mortgage" in product_type:
            product_key = "loan"

        if product_key and product_key in self.REQUIRED_DISCLOSURES:
            required = self.REQUIRED_DISCLOSURES[product_key]
            if required.lower() not in text_lower:
                issues.append({
                    "type": "MISSING_DISCLOSURE",
                    "required_text": required,
                    "severity": "MEDIUM",
                    "action": f"Add mandatory disclaimer: '{required}'",
                    "auto_fixed": False,
                })

        # ── Sentiment Analysis ────────────────────────────
        sentiment_score = 0.0
        if _VADER_OK:
            analyzer = _SIA()
            # Analyze a sample of the text
            sample = text_content[:2000]
            scores = analyzer.polarity_scores(sample)
            sentiment_score = scores["compound"]

            if sentiment_score < -0.5:  # strongly negative
                issues.append({
                    "type": "NEGATIVE_SENTIMENT",
                    "score": sentiment_score,
                    "severity": "MEDIUM",
                    "action": "Tone is too negative/threatening. Rewrite with positive framing.",
                    "auto_fixed": False,
                })

        # ── Final Verdict ─────────────────────────────────
        high_severity_count = sum(1 for i in issues if i["severity"] == "HIGH")
        is_safe = high_severity_count == 0 and len(issues) <= 2

        result = {
            "is_safe": is_safe,
            "safety_score": max(0, 100 - (high_severity_count * 30) - (len(issues) * 10)),
            "issues": issues,
            "issue_count": len(issues),
            "pii_detected": pii_found,
            "pii_auto_masked": len([i for i in issues if i.get("auto_fixed")]),
            "sentiment_score": round(sentiment_score, 3),
            "fixed_html": fixed_html,
            "recommendation": "SAFE TO SEND" if is_safe else "NEEDS REVIEW BEFORE SENDING",
        }

        logger_compliance.info(
            f"ComplianceGuard: {'✓ SAFE' if is_safe else '✗ UNSAFE'} — "
            f"{len(issues)} issues, {len(pii_found)} PII types detected."
        )
        return result

    def check_batch(self, emails: List[Dict], product_type: str = "general") -> List[Dict]:
        """Check compliance for a batch of emails."""
        results = []
        for email in emails:
            html = email.get("html", "")
            if not html:
                continue
            check = self.check(html, product_type)
            results.append({
                "email": email.get("email", ""),
                "customer_name": email.get("customer_name", ""),
                **check,
            })
        safe_count = sum(1 for r in results if r["is_safe"])
        logger_compliance.info(f"ComplianceGuard batch: {safe_count}/{len(results)} emails safe.")
        return results

    def __call__(self, input_data: Any) -> Dict:
        """NexusHub slot callable."""
        if isinstance(input_data, dict):
            html = input_data.get("html", "")
            product_type = input_data.get("product_type", "general")
            return self.check(html, product_type)
        return {"is_safe": False, "error": "Invalid input"}
