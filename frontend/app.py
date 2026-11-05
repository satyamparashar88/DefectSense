
import os
import io
import requests
import streamlit as st
from PIL import Image
import base64

# 1. Application configuration

st.set_page_config(
    page_title="DefectSense",
    page_icon="🔍",
    layout="wide"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

# 2. Application interface

st.title("🔍 DefectSense")
st.subheader("AI-Powered PCB Defect Detection")

st.write(
    "Upload a Printed Circuit Board (PCB) image "
    "to detect and identify possible manufacturing defects."
)

st.divider()

# Upload PCB image

uploaded_file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold

confidence = st.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# 3. Send image to FastAPI

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.subheader("Uploaded PCB Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Defects", type="primary"):

        try:
            with st.spinner("Analyzing PCB image..."):

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    },
                    data={
                        "confidence": confidence
                    },
                    timeout=120
                )

                response.raise_for_status()

                summary = response.json()

            # 4. Display detection results

            st.divider()
            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original PCB Image**")
                st.image(
                    image,
                    use_container_width=True
                )

            with col2:
                st.markdown("**Detected Defects**")

                # The backend currently returns JSON,
                # not an annotated image.
                
                annotated_image_bytes=base64.b64decode(
                    summary["annotated_image"]
                )
                st.image(
                    annotated_image_bytes,
                    caption="PCB Image With Detected Defects",
                    use_container_width=True

                )

            # 5. Display inspection summary

            st.divider()
            st.subheader("Defect Summary")

            total_defects = summary["total_defects"]

            st.metric(
                "Total Defects Detected",
                total_defects
            )

            st.write(
                "**Inspection Status:**",
                summary["inspection_status"]
            )

            st.write(
                "**Average Confidence:**",
                f"{summary['average_confidence']:.2%}"
            )

            if total_defects > 0:

                defect_counts = summary["defect_counts"]

                st.markdown("**Defect Counts**")

                st.dataframe(
                    [
                        {
                            "Defect": defect,
                            "Count": count
                        }
                        for defect, count in defect_counts.items()
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Individual Detections**")

                detected = [
                    {
                        "Defect": item["defect"],
                        "Confidence": f"{item['confidence']:.2%}"
                    }
                    for item in summary["detections"]
                ]

                st.dataframe(
                    detected,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.success(
                    "No defects detected above the selected "
                    "confidence threshold."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. "
                "Please start the backend first."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. "
                "Please try again."
            )

        except requests.exceptions.HTTPError as error:
            st.error(f"Backend error: {error}")

        except Exception as error:
            st.error(f"Detection failed: {error}")

else:
    st.info("Please upload a PCB image to get started.")

st.divider()

st.caption(
    "DefectSense | PCB Defect Detection using RF-DETR Nano"
)
