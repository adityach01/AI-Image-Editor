"""
AI-Powered Image Editing Platform - Week 2
Image Management System + AI Image Editing + Version History
"""
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from utils import ImageManager, GeminiVisionAPI, format_file_size, format_date, PRESET_EDITS

# Gemini model default
GEMINI_MODEL_DEFAULT = "gemini-3.6-flash"

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .image-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        transition: box-shadow 0.3s;
    }
    .image-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .caption-text {
        font-size: 0.9rem;
        color: #555;
        margin-top: 8px;
    }
    .stat-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'library'
if 'selected_image_id' not in st.session_state:
    st.session_state.selected_image_id = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'edit_prompt' not in st.session_state:
    st.session_state.edit_prompt = ""
if 'selected_version' not in st.session_state:
    st.session_state.selected_version = None

# Initialize managers
@st.cache_resource
def get_image_manager():
    return ImageManager()

@st.cache_resource
def get_gemini_api(_api_key, _model_name):
    """
    Get Gemini API instance. Cache depends on API key and model name.
    When either changes in .env, the cache will refresh.
    """
    if not _api_key:
        return None

    return GeminiVisionAPI(api_key=_api_key, model_name=_model_name)


image_manager = get_image_manager()

# Get API key and model name from environment (cache will refresh if either changes)
api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_name = os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT).strip()
gemini_api = get_gemini_api(api_key, model_name)


def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("# 🎨 AI Image Editor")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        if st.button("📚 Image Library", use_container_width=True):
            st.session_state.current_view = 'library'
            st.session_state.selected_image_id = None
            st.rerun()
        
        if st.button("⬆️ Upload Images", use_container_width=True):
            st.session_state.current_view = 'upload'
            st.rerun()
        
        st.markdown("---")
        
        # Stats
        all_images = image_manager.get_all_images()
        st.markdown("### 📊 Statistics")
        st.metric("Total Images", len(all_images))
        
        # Count total edits
        total_edits = sum(len(img.get('versions', [])) for img in all_images)
        if total_edits > 0:
            st.metric("Total Edits", total_edits)
        
        if all_images:
            total_size = sum(img['file_size'] for img in all_images)
            st.metric("Storage Used", format_file_size(total_size))
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔑 API Status")
        if gemini_api:
            st.success("✅ Gemini API Connected")
            current_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT).strip()
            st.caption(f"Model: `{current_model}`")
        else:
            st.error("❌ API Key Not Set")
            st.info("Add your Gemini API key to `.env` file")
            st.markdown("[Get Free API Key](https://makersuite.google.com/app/apikey)")


def render_upload_page():
    """Render the upload page"""
    st.markdown('<div class="main-header">📤 Upload Images</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload images to your library and get AI-generated captions</div>', unsafe_allow_html=True)
    
    if not gemini_api:
        st.warning("⚠️ Gemini API key not configured. Please add your API key to the `.env` file.")
        st.code(
            "GEMINI_API_KEY=your_actual_api_key_here\n"
            "GEMINI_MODEL=gemini-3.6-flash",
            language="bash"
        )
        st.markdown("[Get your free API key here](https://makersuite.google.com/app/apikey)")
        st.info("💡 The GEMINI_MODEL is optional. Default is gemini-3.6-flash")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images to upload",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        help="Supported formats: PNG, JPG, JPEG, WEBP"
    )
    
    if uploaded_files:
        st.markdown(f"### {len(uploaded_files)} file(s) selected")
        
        # Option to auto-generate captions
        auto_caption = st.checkbox("Auto-generate captions with AI", value=True)
        
        if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save image first (without caption)
                temp_metadata = image_manager.save_image(uploaded_file, caption="Processing...")
                
                # Generate caption if enabled
                if auto_caption:
                    with st.spinner(f"Generating caption for {uploaded_file.name}..."):
                        caption = generate_caption_safe(temp_metadata['path'])
                        image_manager.update_caption(temp_metadata['id'], caption)
                else:
                    image_manager.update_caption(temp_metadata['id'], "No caption generated")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Successfully uploaded {len(uploaded_files)} image(s)!")
            st.balloons()
            
            if st.button("Go to Library"):
                st.session_state.current_view = 'library'
                st.rerun()


def render_library_page():
    """Render the image library page"""
    st.markdown('<div class="main-header">📚 Image Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse and search your image collection</div>', unsafe_allow_html=True)
    
    # Search bar
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search images by caption or filename",
            value=st.session_state.search_query,
            placeholder="e.g., 'beach', 'sunset', 'portrait'..."
        )
        st.session_state.search_query = search_query
    
    with col2:
        st.markdown("###")  # Spacing
        if st.button("Clear Search", use_container_width=True):
            st.session_state.search_query = ""
            st.rerun()
    
    # Get images (filtered by search if applicable)
    if search_query:
        images = image_manager.search_images(search_query)
        st.info(f"Found {len(images)} image(s) matching '{search_query}'")
    else:
        images = image_manager.get_all_images()
    
    if not images:
        st.info("📭 No images in your library yet. Upload some to get started!")
        if st.button("Upload Images"):
            st.session_state.current_view = 'upload'
            st.rerun()
        return

    # Display images in grid
    st.markdown(f"### {len(images)} Image(s)")
    
    # Create grid layout (3 columns)
    cols_per_row = 3
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, col in enumerate(cols):
            img_idx = i + col_idx
            if img_idx < len(images):
                image_data = images[img_idx]
                
                with col:
                    # Display image with better error handling
                    image_path = Path(image_data['path'])
                    if image_path.exists():
                        try:
                            img = Image.open(image_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    else:
                        st.error(f"❌ File not found at: {image_path}")
                        st.caption("The image file may have been moved or deleted.")
                    
                    # Image info
                    st.markdown(f"**{image_data['original_name']}**")
                    st.caption(format_date(image_data['upload_date']))
                    
                    # Caption (truncated)
                    caption = image_data['caption']
                    if len(caption) > 100:
                        caption = caption[:100] + "..."
                    st.markdown(f'<div class="caption-text">{caption}</div>', unsafe_allow_html=True)
                    
                    # View details button
                    if st.button("👁️ View Details", key=f"view_{image_data['id']}", use_container_width=True):
                        st.session_state.selected_image_id = image_data['id']
                        st.session_state.current_view = 'detail'
                        st.rerun()
                    
                    st.markdown("---")


def render_detail_page():
    """Render the image detail page"""
    if not st.session_state.selected_image_id:
        st.error("No image selected")
        return
    
    image_data = image_manager.get_image_by_id(st.session_state.selected_image_id)
    
    if not image_data:
        st.error("Image not found")
        return
    
    # Back button
    if st.button("← Back to Library"):
        st.session_state.current_view = 'library'
        st.session_state.selected_image_id = None
        st.rerun()
    
    st.markdown("---")
    
    # Layout: Image on left, details on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {image_data['original_name']}")
        image_path = Path(image_data['path'])
        
        if image_path.exists():
            try:
                img = Image.open(image_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.error(f"❌ Image file not found at: {image_path}")
            st.caption("The image file may have been moved or deleted.")
    
    with col2:
        st.markdown("### 📋 Details")
        
        st.markdown("**Upload Date:**")
        st.text(format_date(image_data['upload_date']))
        
        st.markdown("**File Size:**")
        st.text(format_file_size(image_data['file_size']))
        
        st.markdown("**Dimensions:**")
        if image_data.get('width') and image_data.get('height'):
            st.text(f"{image_data['width']} × {image_data['height']} px")
        else:
            # Fallback: try to get from file
            image_path = Path(image_data['path'])
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    st.text(f"{img.width} × {img.height} px")
                except:
                    st.text("Unknown")
            else:
                st.text("File not found")
        
        st.markdown("---")
        
        st.markdown("### 🤖 AI Caption")
        st.info(image_data['caption'])
        
        # Regenerate caption option
        if gemini_api and image_path.exists() and st.button("🔄 Regenerate Caption", use_container_width=True):
            with st.spinner("Generating new caption..."):
                new_caption = generate_caption_safe(str(image_path))
                image_manager.update_caption(image_data['id'], new_caption)
                st.success("Caption updated!")
                st.rerun()
        
        st.markdown("---")
        
        # Edit with AI button
        st.markdown("### ✨ Edit with AI")
        if gemini_api and image_path.exists():
            if st.button("🎨 Edit This Image", use_container_width=True, type="primary"):
                st.session_state.current_view = 'edit'
                st.rerun()
        else:
            st.info("API key required for editing")
        
        st.markdown("---")
        
        # Version History
        st.markdown("### 📝 Version History")
        versions = image_data.get('versions', [])
        
        if versions:
            st.success(f"{len(versions)} edit(s) made")
            
            # Show versions in reverse order (newest first)
            for idx, version in enumerate(reversed(versions)):
                version_num = version['version_number']
                with st.expander(f"Version {version_num} - {format_date(version['created_date'])}", expanded=(idx == 0)):
                    # Display version thumbnail
                    version_path = Path(version['path'])
                    if version_path.exists():
                        try:
                            version_img = Image.open(version_path)
                            st.image(version_img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading version: {e}")
                    
                    st.caption(f"**Edit:** {version['edit_prompt']}")
                    if version.get('edit_description'):
                        st.caption(f"**How:** {version['edit_description']}")
                    st.caption(f"**Size:** {format_file_size(version['file_size'])}")
        else:
            st.info("No edits yet. Click 'Edit This Image' to start.")
        
        st.markdown("---")
        
        st.markdown("### 🗑️ Delete Image")
        confirm_delete = st.checkbox("Confirm delete this image", key=f"confirm_delete_{image_data['id']}")
        if st.button("Delete This Image", use_container_width=True, disabled=not confirm_delete):
            if delete_image_safe(image_data['id']):
                st.success("Image deleted.")
                st.session_state.current_view = 'library'
                st.session_state.selected_image_id = None
                st.rerun()
            else:
                st.error("Could not delete this image.")


def render_edit_page():
    """Render the AI image editing page"""
    if not st.session_state.selected_image_id:
        st.error("No image selected")
        return
    
    image_data = image_manager.get_image_by_id(st.session_state.selected_image_id)
    
    if not image_data:
        st.error("Image not found")
        return
    
    # Back button
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.current_view = 'detail'
            st.rerun()
    with col_title:
        st.markdown('<div class="main-header">✨ Edit with AI</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Layout: Image on left, edit controls on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {image_data['original_name']}")
        image_path = Path(image_data['path'])
        
        if image_path.exists():
            try:
                img = Image.open(image_path)
                st.image(img, use_container_width=True, caption="Original Image")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
        else:
            st.error(f"❌ Image file not found")
            return
    
    with col2:
        st.markdown("### 🎨 Edit Options")
        
        # Preset edit buttons
        st.markdown("**Quick Edits:**")
        preset_selected = st.selectbox(
            "Choose a preset edit",
            ["Custom..."] + list(PRESET_EDITS.keys()),
            key="preset_selector"
        )
        
        st.markdown("---")
        
        # Natural language prompt
        st.markdown("**Custom Edit Prompt:**")
        
        if preset_selected != "Custom...":
            default_prompt = PRESET_EDITS[preset_selected]
        else:
            default_prompt = st.session_state.edit_prompt
        
        edit_prompt = st.text_area(
            "Describe what you want to change:",
            value=default_prompt,
            height=100,
            placeholder="E.g., 'Remove the person on the left' or 'Make the sky more blue'",
            key="edit_prompt_input"
        )
        
        st.session_state.edit_prompt = edit_prompt
        
        st.markdown("---")
        
        # Process edit button
        if not gemini_api:
            st.error("Gemini API key required")
            return
        
        if st.button("🚀 Apply Edit", use_container_width=True, type="primary", disabled=not edit_prompt.strip()):
            if not edit_prompt.strip():
                st.error("Please enter an edit prompt")
                return
            
            with st.spinner("🤖 AI is analyzing your request..."):
                try:
                    # Step 1: Interpret the edit
                    edit_info = gemini_api.interpret_edit_prompt(edit_prompt)
                    st.session_state.edit_info = edit_info
                    
                    # Step 2: Generate edit instructions
                    instructions = gemini_api.generate_edit_instructions(str(image_path), edit_prompt)
                    
                    # Display what will be done
                    st.info(f"**Edit Type:** {edit_info['edit_type'].replace('_', ' ').title()}")
                    st.info(f"**How it will be done:**\n{instructions}")
                    
                except Exception as e:
                    st.error(f"Error analyzing edit request: {e}")
                    return
            
            # Step 3: Simulate the edit
            with st.spinner("🎨 Applying edit..."):
                try:
                    # Simulate edit (in production, use image-to-image API)
                    edited_img = gemini_api.simulate_edit(str(image_path), edit_info)
                    
                    # Save the edited version
                    version_data = image_manager.save_edited_version(
                        image_data['id'],
                        edited_img,
                        edit_prompt,
                        instructions
                    )
                    
                    if version_data:
                        st.success(f"✅ Edit applied! Version {version_data['version_number']} created.")
                        st.balloons()
                        
                        # Display the edited image
                        st.image(edited_img, caption=f"Edited Version {version_data['version_number']}", use_container_width=True)
                        
                        # Option to continue editing or go back
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("🔄 Edit Again", use_container_width=True):
                                st.session_state.edit_prompt = ""
                                st.rerun()
                        with col_b:
                            if st.button("✅ View Details", use_container_width=True):
                                st.session_state.current_view = 'detail'
                                st.rerun()
                    else:
                        st.error("Failed to save edited version")
                
                except Exception as e:
                    st.error(f"Error applying edit: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("---")
        
        # Help section
        with st.expander("💡 Tips for Better Edits"):
            st.markdown("""
            **Be specific:**
            - ✅ "Remove the tree on the right side"
            - ❌ "Make it better"
            
            **Mention locations:**
            - ✅ "Add a cloud in the top left corner"
            - ❌ "Add a cloud"
            
            **Describe desired outcome:**
            - ✅ "Make the image brighter and more vibrant"
            - ❌ "Fix the lighting"
            
            **Note:** This is a demo using simulated edits. 
            In production, advanced image-to-image generation 
            models would be used for complex edits.
            """)


def generate_caption_safe(image_path: str) -> str:
    """Generate caption with simple error handling."""
    if not gemini_api:
        return "No caption generated"

    try:
        return gemini_api.generate_caption(image_path)
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Caption generation failed"


def delete_image_safe(image_id: str) -> bool:
    """Delete image safely using available ImageManager delete methods."""
    try:
        # Try common method names
        for method_name in ("delete_image", "remove_image", "delete"):
            if hasattr(image_manager, method_name):
                method = getattr(image_manager, method_name)
                result = method(image_id)
                # Treat None as success (common pattern)
                return True if result is None else bool(result)

        st.error("Delete method not found in ImageManager.")
        return False
    except Exception as e:
        st.error(f"Error deleting image: {e}")
        return False


def delete_all_images_safe():
    """Delete all images currently stored in library."""
    images = image_manager.get_all_images()
    deleted_count = 0
    failed_count = 0

    for img in images:
        if delete_image_safe(img["id"]):
            deleted_count += 1
        else:
            failed_count += 1

    # clear selected image if it was deleted
    st.session_state.selected_image_id = None
    return deleted_count, failed_count


def main():
    """Main application"""
    render_sidebar()
    
    # Route to appropriate page
    if st.session_state.current_view == 'upload':
        render_upload_page()
    elif st.session_state.current_view == 'detail':
        render_detail_page()
    elif st.session_state.current_view == 'edit':
        render_edit_page()
    else:  # library
        render_library_page()


if __name__ == "__main__":
    main()
