"""
============================================================
NexusForge — ARViz: AR Email Preview
============================================================
Generates QR code linking to AR.js viewer for email HTML.
Button in UI → scan QR → 3D email pop-up on phone.
============================================================
"""

import base64
import hashlib
import io
import logging
from typing import Optional

logger = logging.getLogger("ARViz")

try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# AR.js viewer HTML template
AR_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NexusForge AR Email Preview</title>
  <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
  <script src="https://raw.githack.com/AR-js-org/AR.js/master/aframe/build/aframe-ar.js"></script>
  <style>body {{ margin: 0; overflow: hidden; background: #000; }}</style>
</head>
<body>
  <a-scene
    embedded
    arjs="sourceType: webcam; debugUIEnabled: false; detectionMode: mono_and_matrix; matrixCodeType: 3x3;"
    renderer="logarithmicDepthBuffer: true; antialias: true;"
    vr-mode-ui="enabled: false"
  >
    <!-- AR Marker: scan this to trigger 3D email -->
    <a-marker preset="hiro">
      <!-- 3D Email Card floating above marker -->
      <a-entity position="0 0.5 0" rotation="-90 0 0">
        <!-- Email card background -->
        <a-plane
          width="2" height="3"
          color="#1e3a6e"
          opacity="0.95"
          shadow="cast: true"
        ></a-plane>

        <!-- Header bar -->
        <a-plane
          position="0 1.2 0.01"
          width="2" height="0.6"
          color="#2563eb"
        ></a-plane>

        <!-- Title text -->
        <a-text
          value="NexusForge\\nBFSI Campaign"
          position="0 1.15 0.02"
          align="center"
          color="white"
          width="3"
          font="dejavu"
        ></a-text>

        <!-- Campaign details -->
        <a-text
          value="{email_subject}"
          position="0 0.4 0.02"
          align="center"
          color="#93c5fd"
          width="2.5"
          wrap-count="25"
        ></a-text>

        <!-- Metric badges -->
        <a-text
          value="Open Rate: 42%\\nClick Rate: 18%\\nStatus: Active"
          position="0 -0.3 0.02"
          align="center"
          color="#a7f3d0"
          width="2.5"
        ></a-text>

        <!-- CTA button -->
        <a-plane
          position="0 -1.0 0.02"
          width="1.2" height="0.35"
          color="#f59e0b"
          animation="property: rotation; to: 0 360 0; loop: true; dur: 8000; easing: linear"
        ></a-plane>
        <a-text
          value="APPLY NOW"
          position="0 -1.0 0.03"
          align="center"
          color="#111"
          width="2"
        ></a-text>

        <!-- Floating particles -->
        <a-entity
          particle-system="preset: snow; color: #60a5fa; particleCount: 30; size: 0.05"
          position="0 0 0"
        ></a-entity>
      </a-entity>

      <!-- Animated ring around card -->
      <a-ring
        color="#f59e0b"
        radius-inner="1.1" radius-outer="1.2"
        rotation="-90 0 0"
        animation="property: rotation; to: -90 360 0; loop: true; dur: 4000; easing: linear"
        opacity="0.7"
      ></a-ring>
    </a-marker>

    <a-entity camera></a-entity>
  </a-scene>

  <!-- Instructions overlay -->
  <div style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.7);
              color:white;padding:12px 24px;border-radius:20px;font-family:Arial;font-size:14px;text-align:center;">
    📱 Point camera at <strong>HIRO marker</strong> to see 3D email preview
  </div>
</body>
</html>"""


class ARViz:
    """
    Generates AR preview for email campaigns.
    Creates a QR code that links to an AR.js web experience.
    Users scan QR → browser opens → point at Hiro marker → 3D email pops up.
    """

    def __init__(self, base_url: str = "https://nexusforge.io/ar"):
        self.base_url = base_url

    def generate_ar_viewer_html(self, email_subject: str = "Exclusive BFSI Offer") -> str:
        """Generate the AR viewer HTML page."""
        return AR_VIEWER_TEMPLATE.format(
            email_subject=email_subject[:40].replace('"', "'")
        )

    def generate_qr_code(self, email_subject: str = "BFSI Campaign") -> Optional[str]:
        """
        Generate a QR code image (base64 PNG) linking to AR viewer.

        Returns:
            Base64-encoded PNG string of the QR code, or None if qrcode not installed.
        """
        if not QR_AVAILABLE:
            logger.warning("ARViz: qrcode not installed. Install with: pip install qrcode[pil]")
            return None

        # Create AR viewer URL with campaign parameters
        campaign_hash = hashlib.md5(email_subject.encode()).hexdigest()[:8]
        ar_url = f"{self.base_url}/preview?id={campaign_hash}&subject={email_subject[:30].replace(' ', '+')}"

        # Generate QR
        qr = qrcode.QRCode(
            version=2,
            error_correction=ERROR_CORRECT_L,
            box_size=8,
            border=3,
        )
        qr.add_data(ar_url)
        qr.make(fit=True)

        # Create styled QR image
        img = qr.make_image(fill_color="#1e3a6e", back_color="white")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        qr_b64 = base64.b64encode(buf.read()).decode("utf-8")

        logger.info(f"ARViz: QR code generated for campaign '{email_subject[:30]}'")
        return qr_b64

    def render_ar_button_streamlit(self, email_subject: str = "Preview Campaign"):
        """Render AR preview button in Streamlit."""
        import streamlit as st

        st.markdown("### 🥽 AR Email Preview")
        st.caption("Scan the QR code with your phone to see a 3D preview of the email")

        qr_b64 = self.generate_qr_code(email_subject)

        if qr_b64:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(
                    f"data:image/png;base64,{qr_b64}",
                    caption="Scan for AR Preview",
                    width=180,
                )
            with col2:
                st.markdown("""
                **How to use:**
                1. 📱 Scan QR with your phone
                2. 🌐 Allow camera access in browser
                3. 🖨️ Point camera at [Hiro AR marker](https://jeromeetienne.github.io/AR.js/data/images/HIRO.jpg)
                4. ✨ Watch the 3D email appear!
                """)

            # Download AR viewer HTML
            ar_html = self.generate_ar_viewer_html(email_subject)
            st.download_button(
                "⬇️ Download AR Viewer (HTML)",
                data=ar_html,
                file_name="nexusforge_ar_preview.html",
                mime="text/html",
            )
        else:
            st.info("Install `qrcode[pil]` to enable AR preview: `pip install qrcode[pil]`")
