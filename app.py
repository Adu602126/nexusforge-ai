"""
============================================================
NexusForge — Main Streamlit Dashboard
============================================================
Pages:
1. 🏠 Dashboard (real-time Plotly metrics)
2. 🚀 Create Campaign (file upload + goal input)
3. ✅ HITL Approvals (approve/reject + blockchain)
4. 💬 AI Chat (LLM-powered campaign queries)
5. 📊 Analytics (detailed campaign analysis)
6. 👥 Team (add members and team name)
============================================================
"""

import json
import os
import sys
import time
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="NexusForge — AI Email Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a6e 50%, #1d4ed8 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    color: white;
  }
  .main-header h1 { margin: 0; font-size: 32px; font-weight: 700; letter-spacing: -0.5px; }
  .main-header p { margin: 4px 0 0 0; color: #93c5fd; font-size: 14px; }

  .metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); }
  .metric-card .metric-value { font-size: 36px; font-weight: 700; color: #1e3a6e; }
  .metric-card .metric-label { font-size: 13px; color: #64748b; margin-top: 4px; }

  .status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
  }
  .status-active { background: #d1fae5; color: #065f46; }
  .status-pending { background: #fef3c7; color: #92400e; }
  .status-error { background: #fee2e2; color: #991b1b; }

  .agent-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    align-items: center;
  }

  .sidebar-logo {
    background: linear-gradient(135deg, #1e3a6e, #2563eb);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    text-align: center;
    color: white;
  }

  div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
  }
  div[data-testid="stButton"] button:hover { transform: translateY(-1px); }

  .campaign-step {
    background: linear-gradient(90deg, #eff6ff, #f8fafc);
    border-left: 4px solid #2563eb;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════════════
def init_session():
    """Initialize session state variables."""
    defaults = {
        "campaign_plan": None,
        "generated_emails": [],
        "sent_results": [],
        "analytics": {},
        "audit_trail": [],
        "chat_history": [],
        "campaign_status": "idle",
        "team_name": "NexusForge Team",
        "team_members": [],
        "nexus_hub": None,
        "blockchain_logger": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ══════════════════════════════════════════════════════════════
# Lazy Imports (with error handling)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_nexus_hub():
    """Load NexusHub and all agents."""
    try:
        from core.nexus_hub import NexusHub
        hub = NexusHub(device="cpu")
        return hub
    except Exception as e:
        st.warning(f"NexusHub load failed: {e}")
        return None

@st.cache_resource
def load_blockchain_logger():
    try:
        from core.hitl_blockchain import BlockchainLogger
        return BlockchainLogger()
    except Exception as e:
        return None


def get_hub():
    if st.session_state.nexus_hub is None:
        st.session_state.nexus_hub = load_nexus_hub()
    return st.session_state.nexus_hub

def get_blockchain():
    if st.session_state.blockchain_logger is None:
        st.session_state.blockchain_logger = load_blockchain_logger()
    return st.session_state.blockchain_logger


# ══════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:28px;">⚡</div>
        <div style="font-weight:700;font-size:18px;">NexusForge</div>
        <div style="font-size:11px;opacity:0.8;">AI Email Automation Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🚀 Create Campaign", "✅ Approvals", "💬 AI Chat", "📊 Analytics", "👥 Team"],
        label_visibility="collapsed",
    )

    st.divider()

    # Agent Status
    st.markdown("**🤖 Agent Status**")
    agents = [
        ("ForgePlanner", "active"),
        ("ContentGen", "active"),
        ("Scheduler", "active"),
        ("Analyzer", "active"),
        ("Optimizer", "active"),
    ]
    for name, status in agents:
        color = "#10b981" if status == "active" else "#f59e0b"
        st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    f'<span style="font-size:13px;">{name}</span>'
                    f'<span style="width:8px;height:8px;background:{color};border-radius:50%;display:inline-block;"></span>'
                    f'</div>', unsafe_allow_html=True)

    st.divider()

    # Team Display
    if st.session_state.team_name:
        st.caption(f"🏢 {st.session_state.team_name}")
    if st.session_state.team_members:
        for m in st.session_state.team_members[:3]:
            st.caption(f"👤 {m}")
        if len(st.session_state.team_members) > 3:
            st.caption(f"...+{len(st.session_state.team_members) - 3} more")


# ══════════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>⚡ NexusForge Dashboard</h1>
        <p>AI-Powered BFSI Email Automation Platform — Real-time Campaign Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "📧", "12,450", "Emails Sent"),
        (col2, "📬", "42.3%", "Open Rate"),
        (col3, "🖱️", "18.7%", "Click Rate"),
        (col4, "⚡", "3.2%", "Conversion"),
        (col5, "💰", "₹2.4L", "Revenue Attribution"),
    ]
    for col, icon, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px;">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Live Charts Row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📈 Campaign Performance (Last 30 Days)")
        try:
            import plotly.graph_objects as go

            # Simulated daily performance data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
            open_rates = 35 + 10 * np.sin(np.linspace(0, 3, 30)) + np.random.normal(0, 2, 30)
            click_rates = 15 + 5 * np.sin(np.linspace(0.5, 3.5, 30)) + np.random.normal(0, 1, 30)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=open_rates, name="Open Rate %",
                                     line=dict(color="#2563eb", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(37,99,235,0.1)"))
            fig.add_trace(go.Scatter(x=dates, y=click_rates, name="Click Rate %",
                                     line=dict(color="#10b981", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=-0.2),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for live charts: `pip install plotly`")

    with col_right:
        st.markdown("#### 🎯 Segment Performance")
        try:
            import plotly.graph_objects as go

            segments = ["Income High", "Income Mid", "High Engagement", "Low Engagement"]
            values = [62, 45, 58, 28]
            colors = ["#1d4ed8", "#3b82f6", "#10b981", "#f59e0b"]

            fig = go.Figure(go.Bar(
                x=values, y=segments,
                orientation="h",
                marker_color=colors,
                text=[f"{v}%" for v in values],
                textposition="inside",
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Open Rate %",
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(range=[0, 80], showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

    # Recent Campaigns
    st.markdown("#### 🗂️ Recent Campaigns")
    campaign_data = {
        "Campaign": ["Home Loan Q1", "Mutual Fund Feb", "Credit Card March", "FD Special", "Insurance Drive"],
        "Sent": [2500, 1800, 3200, 1200, 900],
        "Open Rate": ["44.2%", "38.7%", "41.5%", "35.1%", "29.8%"],
        "Click Rate": ["19.3%", "15.2%", "22.1%", "11.4%", "8.7%"],
        "Status": ["✅ Completed", "✅ Completed", "🔄 Active", "⏸️ Paused", "✅ Completed"],
    }
    df_campaigns = pd.DataFrame(campaign_data)
    st.dataframe(df_campaigns, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: Create Campaign
# ══════════════════════════════════════════════════════════════
elif page == "🚀 Create Campaign":
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Create New Campaign</h1>
        <p>Upload customer data, set your goal, and let NexusForge AI do the rest</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Data Upload
    st.markdown("### Step 1: Upload Customer Data")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Customer CSV",
            type=["csv"],
            help="Required columns: name, email, income, location, past_opens"
        )
        if uploaded_file:
            df_preview = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_preview)} customers")
            st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)

            # Save to temp path
            csv_path = "/tmp/uploaded_customers.csv"
            df_preview.to_csv(csv_path, index=False)
        else:
            csv_path = "data/customers.csv"
            st.info("💡 No file uploaded — will use demo customer data (100 records)")

    with col2:
        st.markdown("""
        **Required CSV Columns:**
        - `name` — Customer name
        - `email` — Email address
        - `income` — Annual income (₹)
        - `location` — City
        - `past_opens` — Historical email opens
        """)
        st.markdown("**Optional:**")
        st.markdown("- `product_interest`, `cibil_score`, `age`")

    st.divider()

    # Step 2: Campaign Goal
    st.markdown("### Step 2: Define Campaign Goal")
    col1, col2 = st.columns(2)
    with col1:
        goal = st.text_area(
            "Campaign Goal",
            value="Promote home loan offers to mid and high income customers in metro cities",
            height=80,
        )
        product_type = st.selectbox(
            "Product Type",
            ["loan_offer", "investment", "credit_card", "insurance", "fd_offer"],
            format_func=lambda x: {
                "loan_offer": "🏠 Home/Personal Loan",
                "investment": "📈 Mutual Fund / Investment",
                "credit_card": "💎 Credit Card",
                "insurance": "🛡️ Insurance",
                "fd_offer": "🏦 Fixed Deposit",
            }[x]
        )
    with col2:
        send_batch_size = st.slider("Emails per batch", 10, 500, 50)
        enable_ab_test = st.checkbox("Enable A/B Testing", value=True)
        enable_compliance = st.checkbox("Run Compliance Check", value=True)
        enable_foresight = st.checkbox("Use Foresight Oracle", value=True)

    st.divider()

    # Step 3: Launch
    st.markdown("### Step 3: Launch Campaign")

    if st.button("⚡ Run NexusForge Campaign", type="primary", use_container_width=True):
        with st.status("🚀 NexusForge is orchestrating your campaign...", expanded=True) as status:

            # ── Step 1: ForgePlanner ──────────────────────
            st.write("🧠 **ForgePlanner**: Analyzing customer data and generating plans...")
            time.sleep(0.5)

            try:
                from agents.forge_planner import ForgePlanner
                planner = ForgePlanner()
                plan = planner.run(csv_path=csv_path, goal=goal)
                st.session_state.campaign_plan = plan
                st.write(f"✅ Plan generated — Variant {plan.get('variant_id', 0)}, "
                         f"Score: {plan.get('predicted_engagement', 0):.1f}/100, "
                         f"Segments: {len(plan.get('segments', []))}")
            except Exception as e:
                st.write(f"⚠️ Planner using fallback mode: {e}")
                # Demo fallback plan
                plan = {
                    "variant_id": 1,
                    "goal": goal,
                    "predicted_engagement": 72.5,
                    "segments": [
                        {"name": "income_high", "size": 25, "email_list": [f"demo{i}@test.com" for i in range(25)]},
                        {"name": "income_mid", "size": 55, "email_list": [f"mid{i}@test.com" for i in range(55)]},
                    ],
                    "outline": {"template_type": product_type, "subject_line": "Special BFSI Offer for You"},
                    "schedule": [{"segment": "income_high", "send_date": "2026-03-10", "send_time": "10:00 AM IST"}],
                    "foresight": {"trend_multiplier": 1.1, "confidence": 0.75},
                }
                st.session_state.campaign_plan = plan

            # ── Step 2: ContentGen ──────────────────────
            st.write("✍️ **ForgeContentGen**: Generating personalized HTML emails...")
            time.sleep(0.5)

            try:
                from agents.forge_content_gen import ForgeContentGen
                content_gen = ForgeContentGen()
                # Use first 10 customers for demo
                demo_customers = [
                    {"name": f"Customer {i}", "email": f"cust{i}@demo.com",
                     "income": random.randint(200000, 1500000), "location": "Mumbai",
                     "past_opens": random.randint(0, 8)}
                    for i in range(10)
                ]
                emails = content_gen.generate(plan=plan, customers=demo_customers)
                st.session_state.generated_emails = emails
                st.write(f"✅ Generated {len([e for e in emails if e.get('html')])} personalized HTML emails")
            except Exception as e:
                st.write(f"⚠️ ContentGen demo mode: {e}")
                emails = [{"email": f"demo{i}@test.com", "html": f"<p>Email {i}</p>", "customer_name": f"Demo {i}"} for i in range(5)]
                st.session_state.generated_emails = emails

            # ── Step 3: Compliance Check ──────────────────
            if enable_compliance and emails:
                st.write("🛡️ **ComplianceGuard**: Scanning for PII and compliance risks...")
                time.sleep(0.3)
                try:
                    from ml.foresight import ComplianceGuard
                    guard = ComplianceGuard()
                    first_html = emails[0].get("html", "") if emails else ""
                    check = guard.check(first_html, product_type)
                    safety_icon = "✅" if check["is_safe"] else "⚠️"
                    st.write(f"{safety_icon} Compliance: {check['recommendation']}, "
                             f"Safety Score: {check['safety_score']}/100, "
                             f"Issues: {check['issue_count']}")
                except Exception as e:
                    st.write(f"⚠️ Compliance check skipped: {e}")

            # ── Step 4: Scheduler ──────────────────────
            st.write(f"📨 **ForgeScheduler**: Sending {min(len(emails), send_batch_size)} emails in batches...")
            time.sleep(0.5)

            try:
                from agents.forge_agents import ForgeScheduler
                scheduler = ForgeScheduler()
                sent = scheduler.send(email_list=emails[:send_batch_size])
                st.session_state.sent_results = sent
                demo_sent = sum(1 for s in sent if "sent" in s.get("status", ""))
                st.write(f"✅ Sent {demo_sent}/{len(sent)} emails (Demo Mode — no real emails sent)")
            except Exception as e:
                st.write(f"⚠️ Scheduler demo mode: {e}")
                st.session_state.sent_results = [{"id": f"demo_{i}", "status": "sent_demo"} for i in range(5)]

            # ── Step 5: Generate Analytics ─────────────────
            st.write("📊 **ForgeAnalyzer**: Computing campaign metrics...")
            time.sleep(0.3)

            # Simulate tracking data
            n_sent = len(st.session_state.sent_results)
            tracking_data = []
            for r in st.session_state.sent_results:
                tracking_data.append({
                    "email_id": r.get("id", ""),
                    "email": r.get("email", ""),
                    "variant": r.get("variant", "A"),
                    "opened": random.random() < 0.42,
                    "clicked": random.random() < 0.18,
                    "bounced": random.random() < 0.03,
                    "unsubscribed": random.random() < 0.01,
                })

            try:
                from agents.forge_agents import ForgeAnalyzer
                analyzer = ForgeAnalyzer()
                analytics = analyzer.analyze(tracking_data)
                st.session_state.analytics = analytics
                st.write(f"✅ Analytics: Open={analytics.get('open_rate', 0):.1f}%, "
                         f"Click={analytics.get('click_rate', 0):.1f}%, "
                         f"Anomalies={analytics.get('anomaly_count', 0)}")
            except Exception as e:
                st.session_state.analytics = {"open_rate": 42.0, "click_rate": 18.0, "bounce_rate": 2.5, "anomaly_count": 0}
                st.write("✅ Analytics computed (demo)")

            st.session_state.campaign_status = "completed"
            status.update(label="✅ Campaign Complete!", state="complete")

        st.success("🎉 Campaign launched successfully! Check Analytics tab for results.")

        # Show plan summary
        if st.session_state.campaign_plan:
            with st.expander("📋 View Campaign Plan (JSON)", expanded=False):
                display_plan = {k: v for k, v in st.session_state.campaign_plan.items() if k != "segments"}
                st.json(display_plan)


# ══════════════════════════════════════════════════════════════
# PAGE 3: HITL Approvals
# ══════════════════════════════════════════════════════════════
elif page == "✅ Approvals":
    st.markdown("""
    <div class="main-header">
        <h1>✅ Human Approval Center</h1>
        <p>Review and approve campaign steps — every decision logged to Ethereum blockchain</p>
    </div>
    """, unsafe_allow_html=True)

    blockchain = get_blockchain()

    if not st.session_state.campaign_plan:
        st.info("💡 Run a campaign first (Create Campaign page) to see approval requests.")
    else:
        # Plan Approval
        st.markdown("### 📋 Campaign Plan Approval")

        plan = st.session_state.campaign_plan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Engagement", f"{plan.get('predicted_engagement', 0):.1f}/100")
        with col2:
            st.metric("Audience Segments", len(plan.get("segments", [])))
        with col3:
            st.metric("Foresight Confidence", f"{plan.get('foresight', {}).get('confidence', 0)*100:.0f}%")

        with st.expander("📊 View Full Plan"):
            st.json({k: v for k, v in plan.items() if k not in ["segments"]})

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve Plan", type="primary", use_container_width=True):
                if blockchain:
                    tx = blockchain.log_approval("campaign_plan", plan, "approved")
                    st.success(f"✅ Approved and logged to blockchain!")
                    st.code(f"TX Hash: {tx['tx_hash'][:50]}\nBlock: {tx['block_number']}\nNetwork: {tx['network']}")
                    st.session_state.audit_trail.append(tx)
                else:
                    st.success("✅ Plan approved (blockchain logging skipped)")
        with col2:
            if st.button("✏️ Revise Plan", use_container_width=True):
                st.warning("✏️ Plan sent back for revision")
        with col3:
            if st.button("❌ Reject Plan", use_container_width=True):
                st.error("❌ Plan rejected")

    st.divider()

    # Voice Approval Demo
    st.markdown("### 🎤 Voice Approval (VoiceApprove)")
    st.caption("Upload a WAV/MP3 recording saying 'yes' or 'no' to approve/reject")

    audio_file = st.file_uploader("Upload voice recording:", type=["wav", "mp3", "m4a"], key="voice_upload")
    if audio_file:
        st.info(f"🎤 Received: {audio_file.name} ({audio_file.size} bytes)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Process Voice (Whisper)", use_container_width=True):
                try:
                    import whisper
                    import tempfile
                    with st.spinner("Transcribing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_file.read())
                        model = whisper.load_model("tiny")
                        result = model.transcribe(tmp.name)
                        transcript = result["text"].lower()
                        if any(w in transcript for w in ["yes", "approve"]):
                            st.success(f"✅ Voice approval: '{transcript}'")
                        elif any(w in transcript for w in ["no", "reject"]):
                            st.error(f"❌ Voice rejection: '{transcript}'")
                        else:
                            st.warning(f"⚠️ Could not parse: '{transcript}'")
                except ImportError:
                    st.warning("Install Whisper: `pip install openai-whisper`")

    st.divider()

    # Audit Trail
    st.markdown("### ⛓️ Blockchain Audit Trail")
    if st.session_state.audit_trail:
        for tx in st.session_state.audit_trail:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin:6px 0;">
                <strong>{tx.get('step', 'Unknown Step')}</strong> — {tx.get('decision', '').upper()}
                <br><code style="font-size:11px;">{tx.get('tx_hash', '')[:60]}...</code>
                <br><span style="color:#64748b;font-size:12px;">Block #{tx.get('block_number', '?')} | {tx.get('network', '')}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No blockchain transactions yet. Approve a campaign step to create an audit trail.")

    # Show blockchain logger history if available
    if blockchain and blockchain.get_audit_trail():
        st.markdown("**All blockchain records:**")
        for tx in blockchain.get_audit_trail():
            st.markdown(f"- `{tx['tx_hash'][:40]}...` — **{tx['step']}** ({tx['decision']})")


# ══════════════════════════════════════════════════════════════
# PAGE 4: AI Chat
# ══════════════════════════════════════════════════════════════
elif page == "💬 AI Chat":
    st.markdown("""
    <div class="main-header">
        <h1>💬 NexusForge AI Assistant</h1>
        <p>Ask anything about your campaigns, analytics, or BFSI email strategy</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about your campaign... (e.g., 'Why is my open rate low?')")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build context from campaign state
                context = ""
                if st.session_state.campaign_plan:
                    context += f"\nCurrent campaign: {st.session_state.campaign_plan.get('goal', '')}"
                    context += f"\nPredicted engagement: {st.session_state.campaign_plan.get('predicted_engagement', 0):.1f}/100"
                if st.session_state.analytics:
                    context += f"\nCurrent metrics: Open={st.session_state.analytics.get('open_rate', 0):.1f}%, Click={st.session_state.analytics.get('click_rate', 0):.1f}%"

                try:
                    import anthropic
                    client = anthropic.Anthropic()
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=600,
                        system=f"""You are NexusForge AI Assistant, an expert in BFSI (Banking, Financial Services, Insurance) 
                        email marketing campaigns. You help marketing teams optimize their email campaigns. 
                        Be concise, actionable, and BFSI-specific.
                        {f'Campaign Context: {context}' if context else ''}""",
                        messages=[{"role": "user", "content": user_input}]
                    )
                    ai_response = response.content[0].text
                except Exception:
                    # Fallback rule-based responses
                    responses = {
                        "open rate": "**To improve open rates:**\n- Test send times (10 AM & 2 PM IST typically best for BFSI)\n- Personalize subject lines with customer name + product benefit\n- A/B test 2-3 subject line variants\n- Clean your email list to reduce bounces",
                        "segment": "**Smart BFSI segmentation:**\n- Income bands: <3L (value), 3-8L (standard), >8L (premium)\n- Engagement: High (5+ opens) vs Low (<3 opens)\n- Product affinity: Loan-seekers vs Investors vs Insurance prospects",
                        "click": "**Boost click rates:**\n- Single, clear CTA button (not multiple links)\n- Mobile-optimized email layout\n- Show personalized loan amount / investment returns\n- Add urgency: 'Offer expires in 3 days'",
                        "bounce": "**Reduce bounce rate:**\n- Validate emails before sending (remove .invalid, double-check)\n- Remove hard bounces immediately\n- Use double opt-in for new subscribers\n- Clean list monthly",
                        "compliance": "**BFSI email compliance:**\n- Always include: 'Investments subject to market risk'\n- Mask PII in all communications\n- Include unsubscribe link\n- Log all sends for audit trail (use our blockchain feature!)",
                    }
                    ai_response = "I'm your NexusForge AI assistant! Ask me about:\n- Improving open/click rates\n- Customer segmentation\n- Compliance requirements\n- Campaign optimization"
                    for keyword, resp in responses.items():
                        if keyword in user_input.lower():
                            ai_response = resp
                            break

                st.write(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    # Quick question chips
    st.markdown("**Quick questions:**")
    quick_q = st.columns(3)
    questions = [
        "Why is my open rate low?",
        "How to segment BFSI customers?",
        "What's the best send time?",
    ]
    for col, q in zip(quick_q, questions):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 5: Analytics
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Campaign Analytics</h1>
        <p>Deep-dive into campaign performance, anomalies, and optimization insights</p>
    </div>
    """, unsafe_allow_html=True)

    analytics = st.session_state.analytics or {
        "total_sent": 100, "opened": 42, "clicked": 18,
        "bounced": 3, "open_rate": 42.0, "click_rate": 18.0,
        "bounce_rate": 3.0, "ctr": 42.9, "anomaly_count": 0,
        "health_score": 87,
    }

    # Metrics
    cols = st.columns(4)
    kpis = [
        ("Total Sent", analytics.get("total_sent", 0), "📧"),
        ("Open Rate", f"{analytics.get('open_rate', 0):.1f}%", "📬"),
        ("Click Rate", f"{analytics.get('click_rate', 0):.1f}%", "🖱️"),
        ("Health Score", f"{analytics.get('health_score', 0)}/100", "💚"),
    ]
    for col, (label, val, icon) in zip(cols, kpis):
        with col:
            st.metric(f"{icon} {label}", val)

    st.divider()

    # Funnel visualization
    try:
        import plotly.graph_objects as go

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 📐 Conversion Funnel")
            total = analytics.get("total_sent", 100)
            opened = analytics.get("opened", 42)
            clicked = analytics.get("clicked", 18)
            fig = go.Figure(go.Funnel(
                y=["Sent", "Opened", "Clicked", "Converted"],
                x=[total, opened, clicked, max(1, int(clicked * 0.18))],
                textinfo="value+percent initial",
                marker={"color": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]},
            ))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("#### 🔘 A/B Test Results")
            fig = go.Figure()
            variants = ["Variant A", "Variant B"]
            open_rates = [42.3, 39.1]
            click_rates = [18.7, 21.2]
            x = np.arange(len(variants))
            fig.add_bar(x=variants, y=open_rates, name="Open Rate %", marker_color="#3b82f6")
            fig.add_bar(x=variants, y=click_rates, name="Click Rate %", marker_color="#10b981")
            fig.update_layout(
                height=300, barmode="group",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for charts")

    # Anomaly alerts
    if analytics.get("anomaly_count", 0) > 0:
        st.markdown("#### ⚠️ Anomalies Detected")
        for anomaly in analytics.get("anomalies", []):
            severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(anomaly.get("severity", "LOW"), "⚪")
            st.warning(f"{severity_color} **{anomaly['type']}**: {anomaly.get('action', '')}")
    else:
        st.success("✅ No anomalies detected — campaign metrics are healthy!")

    # AR Preview
    st.divider()
    st.markdown("#### 🥽 AR Email Preview")
    try:
        from utils.ar_viz import ARViz
        ar = ARViz()
        subject = st.session_state.campaign_plan.get("outline", {}).get("subject_line", "BFSI Offer") if st.session_state.campaign_plan else "BFSI Campaign Preview"
        ar.render_ar_button_streamlit(subject)
    except Exception as e:
        st.info(f"AR preview: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 6: Team
# ══════════════════════════════════════════════════════════════
elif page == "👥 Team":
    st.markdown("""
    <div class="main-header">
        <h1>👥 Team Management</h1>
        <p>Add your team members and team name — displayed throughout the platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🏢 Team Details")

        new_team_name = st.text_input(
            "Team Name",
            value=st.session_state.team_name,
            placeholder="e.g., BFSI Innovation Lab"
        )

        hackathon_name = st.text_input(
            "Hackathon / Event Name",
            placeholder="e.g., FinHack 2026"
        )

        institution = st.text_input(
            "College / Organization",
            placeholder="e.g., IIT Bombay"
        )

        if st.button("💾 Save Team Details", type="primary"):
            st.session_state.team_name = new_team_name
            st.success(f"✅ Team details saved!")

    with col2:
        st.markdown("### 👤 Team Members")
        st.caption("Add member names — they'll appear in the sidebar and reports")

        member_input = st.text_input("Member Name", placeholder="e.g., Priya Sharma")
        member_role = st.selectbox("Role", ["Developer", "ML Engineer", "Designer", "Product Manager", "Domain Expert", "Other"])

        if st.button("➕ Add Member"):
            if member_input.strip():
                display = f"{member_input.strip()} ({member_role})"
                if display not in st.session_state.team_members:
                    st.session_state.team_members.append(display)
                    st.success(f"Added: {display}")
                else:
                    st.warning("Member already added!")

    # Current Team Display
    st.divider()
    st.markdown("### 🏆 Current Team")

    if st.session_state.team_members or st.session_state.team_name:
        # Team card
        members_html = ""
        for i, m in enumerate(st.session_state.team_members):
            name_part = m.split(" (")[0]
            role_part = m.split("(")[1].replace(")", "") if "(" in m else ""
            color = ["#2563eb", "#10b981", "#7c3aed", "#f59e0b", "#ef4444"][i % 5]
            members_html += f"""
            <div style="display:flex;align-items:center;padding:10px;border-bottom:1px solid #f1f5f9;">
                <div style="width:36px;height:36px;background:{color};border-radius:50%;display:flex;
                            align-items:center;justify-content:center;color:white;font-weight:bold;margin-right:12px;">
                    {name_part[0].upper()}
                </div>
                <div>
                    <div style="font-weight:600;font-size:14px;">{name_part}</div>
                    <div style="color:#64748b;font-size:12px;">{role_part}</div>
                </div>
            </div>
            """

        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;max-width:600px;">
            <div style="background:linear-gradient(135deg,#1e3a6e,#2563eb);padding:20px;color:white;">
                <h3 style="margin:0;font-size:20px;">⚡ {st.session_state.team_name}</h3>
                {f'<p style="margin:4px 0 0 0;opacity:0.8;font-size:13px;">{hackathon_name if "hackathon_name" in dir() else ""}</p>' if 'hackathon_name' in locals() and hackathon_name else ''}
                <p style="margin:4px 0 0 0;opacity:0.7;font-size:12px;">{len(st.session_state.team_members)} member(s)</p>
            </div>
            {members_html if members_html else '<div style="padding:20px;color:#94a3b8;">No members added yet</div>'}
        </div>
        """, unsafe_allow_html=True)

        # Remove member option
        if st.session_state.team_members:
            st.divider()
            to_remove = st.selectbox("Remove a member:", ["— select —"] + st.session_state.team_members)
            if st.button("🗑️ Remove Member") and to_remove != "— select —":
                st.session_state.team_members.remove(to_remove)
                st.rerun()
    else:
        st.info("Add your team name and members above to get started!")

    # Export team card
    if st.session_state.team_members:
        st.divider()
        team_data = {
            "team_name": st.session_state.team_name,
            "members": st.session_state.team_members,
        }
        st.download_button(
            "⬇️ Export Team Card (JSON)",
            data=json.dumps(team_data, indent=2),
            file_name="nexusforge_team.json",
            mime="application/json",
        )
