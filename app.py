import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Configure the Streamlit page
st.set_page_config(page_title="Blush & Skin Tone Analyzer", layout="wide")

class SkinToneAnalyzer:
    def __init__(self):
        # Load OpenCV's pre-trained face detector
        # We use the default path included in cv2.data
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Blush Recommendations Database
        self.blush_recommendations = {
            "Fair": {
                "Cool": [
                    {"name": "Soft Rose", "hex": "#F4C2C2", "desc": "A gentle, cool-toned pink."},
                    {"name": "Baby Pink", "hex": "#F8C3CD", "desc": "Light and airy pink."},
                    {"name": "Classic Mauve", "hex": "#E0B0FF", "desc": "A cool purple-pink."}
                ],
                "Warm": [
                    {"name": "Peach", "hex": "#FFDAB9", "desc": "Soft orange-pink warmth."},
                    {"name": "Soft Coral", "hex": "#F88379", "desc": "Vibrant but light."},
                    {"name": "Apricot", "hex": "#FBCEB1", "desc": "Golden undertone enhancer."}
                ],
                "Neutral": [
                    {"name": "Dusty Pink", "hex": "#D8B2BE", "desc": "Muted and natural."},
                    {"name": "Peachy Pink", "hex": "#FF9999", "desc": "Balances warm and cool."}
                ]
            },
            "Medium": {
                "Cool": [
                    {"name": "Berry", "hex": "#990F4B", "desc": "Rich pink with blue undertones."},
                    {"name": "Plum", "hex": "#DDA0DD", "desc": "Cool purple shade."},
                    {"name": "Rosy Pink", "hex": "#FF007F", "desc": "Vibrant pink pop."}
                ],
                "Warm": [
                    {"name": "Rich Coral", "hex": "#FF7F50", "desc": "Punchy orange-pink."},
                    {"name": "Terracotta", "hex": "#E2725B", "desc": "Earthy reddish-brown."},
                    {"name": "Bronze", "hex": "#CD7F32", "desc": "Sun-kissed metallic look."}
                ],
                "Neutral": [
                    {"name": "Mauve", "hex": "#E0B0FF", "desc": "Universally flattering purple-pink."},
                    {"name": "Rosewood", "hex": "#9E4244", "desc": "Deep, natural flush."}
                ]
            },
            "Deep": {
                "Cool": [
                    {"name": "Deep Fuchsia", "hex": "#C154C1", "desc": "Bright cool pink."},
                    {"name": "Dark Berry", "hex": "#4B0082", "desc": "Intense purple."},
                    {"name": "Raisin", "hex": "#290908", "desc": "Deep brownish-red."}
                ],
                "Warm": [
                    {"name": "Brick Red", "hex": "#CB4154", "desc": "Warm, earthy red."},
                    {"name": "Tangerine", "hex": "#F28500", "desc": "Bold orange."},
                    {"name": "Burnt Orange", "hex": "#CC5500", "desc": "Deep warmth."}
                ],
                "Neutral": [
                    {"name": "Rich Mauve", "hex": "#800080", "desc": "Deep purple pink."},
                    {"name": "Deep Rose", "hex": "#C32148", "desc": "Classic red-pink."}
                ]
            }
        }

    def process_image(self, image_array):
        """
        Processes the image array (RGB) to detect skin tone using Haar Cascades.
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None

        # Process the largest face found
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]

        # --- Geometric Estimation of Cheek Areas ---
        # Cheeks are generally located in the middle vertical third, 
        # and the outer horizontal quarters of the face box.
        
        # Define Regions of Interest (ROIs) relative to face box
        # Left Cheek (Viewer's Left)
        lc_x1, lc_x2 = x + int(0.15 * w), x + int(0.35 * w)
        lc_y1, lc_y2 = y + int(0.45 * h), y + int(0.65 * h)
        
        # Right Cheek (Viewer's Right)
        rc_x1, rc_x2 = x + int(0.65 * w), x + int(0.85 * w)
        rc_y1, rc_y2 = y + int(0.45 * h), y + int(0.65 * h)
        
        # Crop crops
        left_cheek = image_array[lc_y1:lc_y2, lc_x1:lc_x2]
        right_cheek = image_array[rc_y1:rc_y2, rc_x1:rc_x2]
        
        # Combine samples
        if left_cheek.size > 0 and right_cheek.size > 0:
            # Stack pixels vertically
            skin_pixels = np.vstack((left_cheek.reshape(-1, 3), right_cheek.reshape(-1, 3)))
        elif left_cheek.size > 0:
            skin_pixels = left_cheek.reshape(-1, 3)
        elif right_cheek.size > 0:
            skin_pixels = right_cheek.reshape(-1, 3)
        else:
            # Fallback to center patch if geometry fails
            center_x, center_y = x + w//2, y + h//2
            skin_pixels = image_array[center_y-10:center_y+10, center_x-10:center_x+10].reshape(-1, 3)

        return skin_pixels

    def get_dominant_color(self, pixels):
        if len(pixels) == 0:
            return np.array([200, 150, 130]) # Default fallback
            
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[0].astype(int)

    def determine_tone_and_undertone(self, rgb_color):
        # Normalize and convert to LAB
        pixel_norm = np.array([[rgb_color / 255.0]], dtype=np.float32)
        lab_color = cv2.cvtColor(pixel_norm, cv2.COLOR_RGB2Lab)[0][0]
        L, a, b = lab_color[0], lab_color[1], lab_color[2]
        
        # Tone (Lightness)
        if L > 65: tone = "Fair"
        elif L > 45: tone = "Medium"
        else: tone = "Deep"

        # Undertone (b channel: yellow vs blue)
        # OpenCV LAB b-channel: <128 is cool, >128 is warm (approx)
        if b > 133: undertone = "Warm"
        elif b < 123: undertone = "Cool"
        else: undertone = "Neutral"

        return tone, undertone

# --- Streamlit UI ---

st.title("✨ Personal Blush Color Analyzer")
st.markdown("Upload a selfie to discover your skin tone and the perfect **Blush** (or 'Bluck') colors for you.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    # Fix orientation if needed by converting properly
    image_array = np.array(image.convert('RGB'))
    
    st.image(image, caption='Uploaded Image', width=300)
    
    with st.spinner('Analyzing skin tone...'):
        analyzer = SkinToneAnalyzer()
        skin_pixels = analyzer.process_image(image_array)
        
        if skin_pixels is not None:
            dominant_skin = analyzer.get_dominant_color(skin_pixels)
            tone, undertone = analyzer.determine_tone_and_undertone(dominant_skin)
            
            # Results Container
            st.divider()
            st.header("Your Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Skin Tone")
                st.info(f"**{tone}** with **{undertone}** undertones.")
                
                # Display skin color swatch
                hex_skin = '#{:02x}{:02x}{:02x}'.format(*dominant_skin)
                st.markdown(f"""
                    <div style='
                        background-color: {hex_skin}; 
                        width: 100px; 
                        height: 100px; 
                        border-radius: 50%; 
                        border: 2px solid #ddd;
                        margin-bottom: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    </div>
                    <p>Approximate Skin Color: {hex_skin}</p>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("Recommended Blush Colors")
                recs = analyzer.blush_recommendations.get(tone, {}).get(undertone, [])
                
                if recs:
                    for rec in recs:
                        # Create a color card for each recommendation
                        st.markdown(f"""
                        <div style="
                            display: flex; 
                            align_items: center; 
                            padding: 10px; 
                            background-color: #f9f9f9; 
                            border-radius: 10px; 
                            margin-bottom: 10px;
                            border: 1px solid #eee;">
                            <div style="
                                width: 50px; 
                                height: 50px; 
                                background-color: {rec['hex']}; 
                                border-radius: 8px; 
                                margin-right: 15px;
                                border: 1px solid #ccc;">
                            </div>
                            <div>
                                <h4 style="margin: 0; color: #333;">{rec['name']}</h4>
                                <p style="margin: 0; font-size: 12px; color: #666;">{rec['hex']} • {rec['desc']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No specific recommendations found for this combination.")
        else:
            st.error("Could not detect a face. Please ensure the face is clearly visible and try again.")

# Footer
st.markdown("---")
st.caption("Analysis based on automated cheek area detection and LAB color space logic.")
