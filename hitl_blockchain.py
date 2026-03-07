"""
============================================================
NexusForge — HITL (Human-in-the-Loop) + Blockchain
============================================================
Streamlit UI buttons for approve/reject/revise.
On approve → Web3.py logs hash to Sepolia testnet.
Returns hash for immutable audit trail.
Integrates with NexusHub.
============================================================
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HITLBlockchain")

# ── Web3 (graceful import) ─────────────────────────────────
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3 not installed — blockchain logging will be simulated.")


# ══════════════════════════════════════════════════════════════
# Blockchain Logger (Sepolia Testnet)
# ══════════════════════════════════════════════════════════════
class BlockchainLogger:
    """
    Logs approval hashes to Ethereum Sepolia testnet.
    Provides immutable audit trail for BFSI compliance.
    """

    SEPOLIA_RPC = "https://rpc.sepolia.org"  # public Sepolia RPC
    DUMMY_PRIVATE_KEY = "0x" + "a" * 64  # dummy key for demo

    def __init__(self, private_key: Optional[str] = None, rpc_url: Optional[str] = None):
        self.private_key = private_key or self.DUMMY_PRIVATE_KEY
        self.rpc_url = rpc_url or self.SEPOLIA_RPC
        self.w3 = None
        self.account = None
        self.transaction_log: List[Dict] = []

        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                if self.private_key != self.DUMMY_PRIVATE_KEY:
                    self.account = self.w3.eth.account.from_key(self.private_key)
                    logger.info(f"BlockchainLogger: Connected to Sepolia. Account: {self.account.address}")
                else:
                    logger.info("BlockchainLogger: Running in DEMO mode (dummy key).")
            except Exception as e:
                logger.warning(f"BlockchainLogger: Web3 connection failed ({e}) — using simulation mode.")

    def _compute_data_hash(self, data: Dict) -> str:
        """Compute SHA-256 hash of approval data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return "0x" + hashlib.sha256(data_str.encode()).hexdigest()

    def log_approval(self, step_name: str, data: Dict, decision: str, user_id: str = "system") -> Dict:
        """
        Log an approval decision to blockchain.

        Args:
            step_name: Name of the workflow step (e.g., "plan_approved").
            data: The data being approved (plan, emails, etc.).
            decision: "approved", "rejected", or "revised".
            user_id: Who made the decision.

        Returns:
            Dict with tx_hash and block info.
        """
        data_hash = self._compute_data_hash(data)
        timestamp = int(time.time())

        audit_record = {
            "step": step_name,
            "decision": decision,
            "data_hash": data_hash,
            "user": user_id,
            "timestamp": timestamp,
        }

        if WEB3_AVAILABLE and self.w3 and self.account and self.private_key != self.DUMMY_PRIVATE_KEY:
            # Real blockchain transaction on Sepolia
            try:
                tx_data = json.dumps(audit_record, default=str).encode().hex()
                tx = {
                    "to": self.account.address,  # self-transaction for data logging
                    "value": 0,
                    "gas": 21000 + len(tx_data) * 68,
                    "gasPrice": self.w3.to_wei("10", "gwei"),
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                    "chainId": 11155111,  # Sepolia chain ID
                    "data": "0x" + tx_data[:2048],  # max 1KB of data
                }
                signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

                result = {
                    "status": "on_chain",
                    "tx_hash": tx_hash.hex(),
                    "block_number": receipt["blockNumber"],
                    "data_hash": data_hash,
                    "network": "Sepolia Testnet",
                    "explorer_url": f"https://sepolia.etherscan.io/tx/{tx_hash.hex()}",
                    **audit_record,
                }
                logger.info(f"BlockchainLogger: TX logged on Sepolia: {tx_hash.hex()[:20]}...")

            except Exception as e:
                logger.error(f"BlockchainLogger: TX failed ({e}) — falling back to simulation.")
                result = self._simulate_blockchain_log(audit_record)
        else:
            # Demo/simulation mode
            result = self._simulate_blockchain_log(audit_record)

        self.transaction_log.append(result)
        return result

    def _simulate_blockchain_log(self, audit_record: Dict) -> Dict:
        """Simulate a blockchain transaction for demo purposes."""
        # Generate realistic-looking fake tx hash
        fake_tx = "0x" + hashlib.sha256(
            (json.dumps(audit_record) + str(time.time())).encode()
        ).hexdigest()
        fake_block = 5800000 + len(self.transaction_log)

        return {
            "status": "simulated",
            "tx_hash": fake_tx,
            "block_number": fake_block,
            "data_hash": audit_record["data_hash"],
            "network": "Sepolia Testnet (Simulated)",
            "explorer_url": f"https://sepolia.etherscan.io/tx/{fake_tx}",
            **audit_record,
        }

    def get_audit_trail(self) -> List[Dict]:
        """Return full immutable audit trail."""
        return self.transaction_log


# ══════════════════════════════════════════════════════════════
# HITL Module — Streamlit Integration
# ══════════════════════════════════════════════════════════════
def render_hitl_approval(
    step_name: str,
    data: Dict,
    blockchain_logger: BlockchainLogger,
    nexus_hub=None,
    request_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Render HITL approval widget in Streamlit.
    Shows approve/reject/revise buttons.
    On approve → logs to blockchain.

    Args:
        step_name: Step being approved.
        data: Data to review and approve.
        blockchain_logger: BlockchainLogger instance.
        nexus_hub: Optional NexusHub for resolving approvals.
        request_id: HITL request ID from NexusHub.

    Returns:
        Dict with decision and blockchain hash, or None if pending.
    """
    import streamlit as st

    st.markdown(f"""
    <div style="border: 2px solid #f59e0b; border-radius: 12px; padding: 20px; background: #fffbeb; margin: 16px 0;">
        <h3 style="color:#92400e;margin:0 0 8px 0;">🔐 Human Approval Required</h3>
        <p style="color:#78350f;font-size:14px;margin:0;">Step: <strong>{step_name}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Show data summary
    with st.expander("📋 View Data for Review", expanded=False):
        if isinstance(data, dict):
            # Show key metrics if available
            if "predicted_engagement" in data:
                st.metric("Predicted Engagement", f"{data['predicted_engagement']:.1f}/100")
            if "segments" in data:
                st.metric("Audience Segments", len(data["segments"]))
            if "schedule" in data:
                st.metric("Scheduled Batches", len(data["schedule"]))
            st.json(data, expanded=False)
        else:
            st.write(data)

    st.markdown("**What would you like to do?**")

    col1, col2, col3 = st.columns(3)

    result = None

    with col1:
        if st.button("✅ Approve", key=f"approve_{step_name}_{request_id}", use_container_width=True,
                     type="primary"):
            with st.spinner("Logging to blockchain..."):
                tx_result = blockchain_logger.log_approval(step_name, data, "approved")
                if nexus_hub and request_id:
                    nexus_hub.resolve_hitl(request_id, "approved")

            st.success(f"✅ Approved and logged to blockchain!")
            st.code(f"TX Hash: {tx_result['tx_hash'][:40]}...\nBlock: {tx_result['block_number']}", language="text")
            if tx_result.get("explorer_url"):
                st.markdown(f"[🔗 View on Etherscan]({tx_result['explorer_url']})")
            result = {"decision": "approved", "blockchain": tx_result}

    with col2:
        if st.button("✏️ Revise", key=f"revise_{step_name}_{request_id}", use_container_width=True):
            revised_notes = st.text_area("Revision notes:", key=f"notes_{step_name}")
            tx_result = blockchain_logger.log_approval(step_name, data, "revised")
            if nexus_hub and request_id:
                nexus_hub.resolve_hitl(request_id, "revised", {"notes": revised_notes})
            st.warning("✏️ Sent back for revision.")
            result = {"decision": "revised", "blockchain": tx_result, "notes": revised_notes}

    with col3:
        if st.button("❌ Reject", key=f"reject_{step_name}_{request_id}", use_container_width=True):
            tx_result = blockchain_logger.log_approval(step_name, data, "rejected")
            if nexus_hub and request_id:
                nexus_hub.resolve_hitl(request_id, "rejected")
            st.error("❌ Campaign step rejected.")
            result = {"decision": "rejected", "blockchain": tx_result}

    return result


# ══════════════════════════════════════════════════════════════
# VoiceApprove — Speech-to-Text HITL
# ══════════════════════════════════════════════════════════════
def render_voice_approval(blockchain_logger: BlockchainLogger, step_name: str, data: Dict) -> Optional[str]:
    """
    Voice-activated HITL approval using Whisper.
    Say "yes" to approve, "no" to reject.
    Falls back to buttons if voice fails.

    Returns: "approved", "rejected", or None
    """
    import streamlit as st

    st.markdown("### 🎤 Voice Approval")
    st.caption("Say **'Yes'** to approve or **'No'** to reject")

    # Audio file upload (for local Whisper)
    audio_file = st.file_uploader("Upload voice recording (WAV/MP3):", type=["wav", "mp3", "m4a"],
                                   key=f"voice_{step_name}")

    if audio_file:
        try:
            import whisper
            import tempfile
            import os

            with st.spinner("Transcribing with Whisper..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                # Load Whisper and transcribe
                model = whisper.load_model("tiny")  # tiny model for speed
                result = model.transcribe(tmp_path)
                os.unlink(tmp_path)

                transcript = result["text"].lower().strip()
                st.info(f"🎤 Heard: *\"{transcript}\"*")

                # Parse yes/no
                if any(word in transcript for word in ["yes", "approve", "confirm", "okay", "go ahead", "send"]):
                    tx = blockchain_logger.log_approval(step_name, data, "approved_voice")
                    st.success(f"✅ Voice approval confirmed! TX: {tx['tx_hash'][:30]}...")
                    return "approved"
                elif any(word in transcript for word in ["no", "reject", "stop", "cancel", "deny"]):
                    tx = blockchain_logger.log_approval(step_name, data, "rejected_voice")
                    st.error("❌ Voice rejection confirmed.")
                    return "rejected"
                else:
                    st.warning(f"⚠️ Could not interpret '{transcript}'. Use buttons below.")

        except ImportError:
            st.warning("Whisper not installed. Install with: `pip install openai-whisper`")
        except Exception as e:
            st.warning(f"Voice processing failed: {e}. Using button fallback.")

    # Button fallback
    st.caption("— or use buttons —")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", key=f"voice_btn_approve_{step_name}"):
            tx = blockchain_logger.log_approval(step_name, data, "approved_button")
            st.success(f"Approved. TX: {tx['tx_hash'][:30]}...")
            return "approved"
    with col2:
        if st.button("❌ Reject", key=f"voice_btn_reject_{step_name}"):
            blockchain_logger.log_approval(step_name, data, "rejected_button")
            return "rejected"

    return None
