import streamlit as st
import numpy as np
from PIL import Image
import cv2

def main():
    st.set_page_config(page_title="Camera & Shapes Input", page_icon="📷")
    st.markdown("## 📷 Input Options: Camera, Mobile & Shapes")
    st.markdown("Capture from camera, use mobile camera, or test with Zig Zag / Dots.")
    
    tabs = st.tabs(["📷 Camera", "📱 Mobile Cam", "〰️ Zig Zag", "⏺️ Dots"])
    
    input_image = None
    input_type = None

    # ── TAB 1: Camera ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('### 📷 Capture from Camera')
        st.caption("Take a photo using your device camera.")
        camera_photo = st.camera_input("Capture a photo", key="camera_capture")
        if camera_photo is not None:
            cam_pil = Image.open(camera_photo).convert("RGB")
            
            # Rotation
            cam_rotation = st.select_slider(
                "🔄 Rotate photo",
                options=[0, 90, 180, 270], value=0, key="cam_rot",
            )
            if cam_rotation != 0:
                cam_pil = cam_pil.rotate(-cam_rotation, expand=True)

            # Cropping
            enable_crop = st.checkbox("Enable cropping", value=False, key="cam_crop_on")
            if enable_crop:
                try:
                    from streamlit_cropper import st_cropper
                    cropped_pil = st_cropper(cam_pil, realtime_update=True, box_color="#2563EB", aspect_ratio=None, key="cam_cropper")
                    if cropped_pil is not None:
                        cam_pil = cropped_pil
                except ImportError:
                    st.warning("Please install `streamlit-cropper` for cropping functionality.")
                    
            st.image(cam_pil, caption="Captured Photo", width=280)
            input_image = np.array(cam_pil)[:, :, ::-1] # Convert to BGR
            input_type = "camera"

    # ── TAB 2: Mobile Cam ──────────────────────────────────────────
    with tabs[1]:
        st.markdown('### 📱 Capture from Mobile')
        st.caption("Scan the QR code with your phone to take a photo. It will appear here.")
        
        import uuid, socket, qrcode
        from pathlib import Path
        
        if "mobile_session_id" not in st.session_state:
            st.session_state["mobile_session_id"] = str(uuid.uuid4())
        sess_id = st.session_state["mobile_session_id"]
        
        def get_local_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()
            return ip

        qr_url = f"http://{get_local_ip()}:8501/?mobile_upload={sess_id}"
        
        qr = qrcode.QRCode(version=1, box_size=5, border=1)
        qr.add_data(qr_url)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white")
        
        st.image(img_qr.get_image(), caption=f"Scan with phone", width=150)
        
        # In a real app this directory needs to exist
        Path("backend/mobile_uploads").mkdir(parents=True, exist_ok=True)
        save_path = Path("backend/mobile_uploads") / f"{sess_id}.png"
        
        if st.button("🔄 Check for Uploaded Photo", key="check_mob"):
            pass
        
        if save_path.exists():
            mob_pil = Image.open(save_path).convert("RGB")
            
            mob_rotation = st.select_slider(
                "🔄 Rotate photo",
                options=[0, 90, 180, 270], value=0, key="mob_rot",
            )
            if mob_rotation != 0:
                mob_pil = mob_pil.rotate(-mob_rotation, expand=True)

            enable_mob_crop = st.checkbox("Enable cropping", value=False, key="mob_crop_on")
            if enable_mob_crop:
                try:
                    from streamlit_cropper import st_cropper
                    cropped_mob_pil = st_cropper(mob_pil, realtime_update=True, box_color="#2563EB", aspect_ratio=None, key="mob_cropper")
                    if cropped_mob_pil is not None:
                        mob_pil = cropped_mob_pil
                except ImportError:
                    st.warning("Please install `streamlit-cropper` for cropping functionality.")

            st.image(mob_pil, caption="Mobile Photo", width=280)
            input_image = np.array(mob_pil)[:, :, ::-1] # Convert to BGR
            input_type = "mobile_camera"

    # ── TAB 3: Zig Zag ─────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 〰️ Zig Zag Test")
        st.caption("Draw a zig zag line or use the auto-generated zig zag.")
        
        zig_zag_mode = st.radio("Mode", ["Auto-Generate", "Draw Custom"], key="zz_mode")
        if zig_zag_mode == "Draw Custom":
            try:
                from streamlit_drawable_canvas import st_canvas
                brush = st.slider("Brush size", 8, 30, 16, key="zz_brush")
                canvas_zz = st_canvas(
                    fill_color="rgba(255,255,255,0)", stroke_width=brush,
                    stroke_color="#FFFFFF", background_color="#0F172A",
                    height=280, width=280, drawing_mode="line", key="canvas_zz",
                )
                if canvas_zz.image_data is not None:
                    arr = canvas_zz.image_data.astype(np.uint8)
                    if arr.max() > 0:
                        input_image = arr
                        input_type = "zig_zag_drawn"
            except ImportError:
                st.error("Please install `streamlit-drawable-canvas` to use the drawing tools.")
        else:
            if st.button("Generate Zig Zag Image"):
                # Create a 280x280 black image with a white zig zag
                img = np.zeros((280, 280, 3), dtype=np.uint8)
                pts = np.array([[40, 40], [140, 140], [40, 240], [140, 280], [240, 140], [240, 40]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 255), thickness=16)
                
                st.session_state["auto_zz"] = img
                
            if "auto_zz" in st.session_state:
                st.image(st.session_state["auto_zz"], caption="Auto-Generated Zig Zag", width=280)
                input_image = st.session_state["auto_zz"]
                input_type = "zig_zag_auto"

    # ── TAB 4: Dots ────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### ⏺️ Dots Test")
        st.caption("Draw dots or use auto-generated dots.")
        
        dots_mode = st.radio("Mode", ["Auto-Generate", "Draw Custom"], key="dots_mode")
        if dots_mode == "Draw Custom":
            try:
                from streamlit_drawable_canvas import st_canvas
                brush2 = st.slider("Dot size (brush)", 8, 30, 16, key="dots_brush")
                canvas_dots = st_canvas(
                    fill_color="rgba(255,255,255,0)", stroke_width=brush2,
                    stroke_color="#FFFFFF", background_color="#0F172A",
                    height=280, width=280, drawing_mode="circle", key="canvas_dots",
                )
                if canvas_dots.image_data is not None:
                    arr = canvas_dots.image_data.astype(np.uint8)
                    if arr.max() > 0:
                        input_image = arr
                        input_type = "dots_drawn"
            except ImportError:
                st.error("Please install `streamlit-drawable-canvas` to use the drawing tools.")
        else:
            if st.button("Generate Dots Image"):
                img = np.zeros((280, 280, 3), dtype=np.uint8)
                cv2.circle(img, (70, 70), 20, (255, 255, 255), -1)
                cv2.circle(img, (210, 70), 20, (255, 255, 255), -1)
                cv2.circle(img, (140, 210), 20, (255, 255, 255), -1)
                cv2.circle(img, (70, 210), 10, (255, 255, 255), -1)
                cv2.circle(img, (210, 210), 10, (255, 255, 255), -1)
                
                st.session_state["auto_dots"] = img
                
            if "auto_dots" in st.session_state:
                st.image(st.session_state["auto_dots"], caption="Auto-Generated Dots", width=280)
                input_image = st.session_state["auto_dots"]
                input_type = "dots_auto"

    st.divider()
    if input_image is not None:
        st.success(f"✅ Input received! Type: `{input_type}`")
        st.image(input_image, caption="Ready for prediction (array shape: {})".format(input_image.shape), width=150)
    else:
        st.info("Select an input method above to begin.")

if __name__ == "__main__":
    main()
