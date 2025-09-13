import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
from streamlit_option_menu import option_menu

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="معالجة الصور التفاعلية",
    page_icon="🖼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص التصميم باستخدام CSS مدمج
st.markdown("""
<style>
    .stApp {
        background-color: #1A4A81;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    h2, h3, h4 {
        color: #2E7D32;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSlider>div>div>div {
        background: #4CAF50;
    }
    .stCheckbox>label {
        color: #2E7D32;
        font-weight: bold;
    }
    .stTextInput>label {
        color: #2E7D32;
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        background-color: #E8F5E9;
        color: #2E7D32;
        font-weight: bold;
    }
    .camera-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# عنوان التطبيق
st.title("🖼 معالجة الصور التفاعلية")
st.markdown("---")

# شريط جانبي للتنقل بين الوحدات
with st.sidebar:
    selected = option_menu(
        menu_title="الوحدات التعليمية",
        options=[
            "مدخل ومعمارية الصور",
            "أنظمة الألوان",
            "العمليات على البكسل",
            "الفلاتر والالتفاف",
            "إزالة الضوضاء",
            "كشف الحواف",
            
            "التحويلات الهندسية",
            "المرشحات المخصصة",
            "تحويل المنظور",
           
            "تحليل الحدود",
            "عتبة متقدمة",
            "المعالجة بالكاميرا",
            "المشروع الختامي"
        ],
        icons=[
            "house", "palette", "sliders", "filter", "ear", 
            "vector-pen", "circle", "arrow-left-right", 
            "stars", "aspect-ratio", "fingerprint", 
            "bounding-box", "sliders2", "camera", "code"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1A4A81"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "right", "margin": "0px"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

# وظيفة لتحميل الصورة
def load_image():
    # خيارات تحميل الصورة
    option = st.radio("اختر طريقة تحميل الصورة:", ("رفع صورة", "استخدام صورة افتراضية"))
    
    img = None
    if option == "رفع صورة":
        uploaded_file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.success("تم تحميل الصورة بنجاح!")
    
    else:  # صورة افتراضية
        default_option = st.selectbox("اختر صورة افتراضية:", 
                                    ("لينا", "بابون", "منظر طبيعي"))
        
        if default_option == "لينا":
            # إنشاء صورة لينا افتراضية (شبكة ألوان)
            img_array = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(img_array, (0, 0), (256, 256), (255, 0, 0), -1)  # أحمر
            cv2.rectangle(img_array, (256, 0), (512, 256), (0, 255, 0), -1)  # أخضر
            cv2.rectangle(img_array, (0, 256), (256, 512), (0, 0, 255), -1)  # أزرق
            cv2.rectangle(img_array, (256, 256), (512, 512), (255, 255, 255), -1)  # أبيض
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            
        elif default_option == "بابون":
            # إنشاء صورة بابون افتراضية (تدرج رمادي)
            img_array = np.zeros((512, 512), dtype=np.uint8)
            for i in range(512):
                img_array[i, :] = i // 2
            img = Image.fromarray(img_array)
            
        else:
            # إنشاء منظر طبيعي افتراضي (تدرج ألوان)
            img_array = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(512):
                # سماء زرقاء
                img_array[i, :, 0] = 255 - i // 2  # أزرق
                img_array[i, :, 1] = 200 - i // 3  # أخضر
                img_array[i, :, 2] = 150 - i // 4  # أحمر
                
                # أرض خضراء
                if i > 400:
                    img_array[i, :, 0] = 50  # أزرق
                    img_array[i, :, 1] = 200  # أخضر
                    img_array[i, :, 2] = 50  # أحمر
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    return img

# وظيفة لعرض معلومات الصورة
def display_image_info(img):
    if img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("معلومات الصورة")
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    height, width, channels = img.shape
                    st.write(f"*الأبعاد:* {width} × {height} بكسل")
                    st.write(f"*عدد القنوات:* {channels}")
                    st.write(f"نوع الصوره :*{type}")
                else:
                    height, width = img.shape
                    st.write(f"*الأبعاد:* {width} × {height} بكسل")
                    st.write(f"*عدد القنوات:* 1 (تدرج رمادي)")
            else:
                st.write(f"*الأبعاد:* {img.width} × {img.height} بكسل")
                st.write(f"*الوضع:* {img.mode}")

# ==============================================================
# الوحدة 1: مدخل ومعمارية الصور الرقمية
# ==============================================================
if selected == "مدخل ومعمارية الصور":
    st.header("مدخل ومعمارية الصور الرقمية")
    
    with st.expander("النظرية"):
        st.markdown("""
        *الصورة الرقمية* هي تمثيل رقمي للصورة المرئية، تتكون من مصفوفة من البكسلات.
        - *البكسل*: أصغر عنصر في الصورة
        - *الأبعاد*: عدد البكسلات في العرض والارتفاع
        - *القنوات*: مكونات الألوان (RGB, Grayscale, إلخ)
        - *العمق اللوني*: عدد البتات المستخدمة لتمثيل لون كل بكسل
        """)
    
    st.subheader("التطبيق العملي")
    
    img = load_image()
    
    if img is not None:
        display_image_info(img)

# ==============================================================
# الوحدة 2: أنظمة الألوان
# ==============================================================
elif selected == "أنظمة الألوان":
    st.header("أنظمة الألوان (Color Spaces)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *أنظمة الألوان* هي طرق مختلفة لتمثيل الألوان:
        - *RGB*: أحمر، أخضر، أزرق (للعرض)
        - *BGR*: أزرق، أخضر، أحمر (لـ OpenCV)
        - *Grayscale*: تدرجات الرمادي
        - *HSV*: صبغة، تشبع، قيمة (لفصل الألوان)
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            color_space = st.selectbox("اختر نظام الألوان:", ("RGB", "Grayscale", "HSV"))
            
            if color_space == "RGB":
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_column_width=True, caption="صورة RGB")
                
            elif color_space == "Grayscale":
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                st.image(gray, use_column_width=True, caption="صورة Grayscale", clamp=True)
            
            else:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
                st.image(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), use_column_width=True, caption="صورة HSV")

# ==============================================================
# الوحدة 3: العمليات على البكسل
# ==============================================================
elif selected == "العمليات على البكسل":
    st.header("العمليات على البكسل (Point Operations)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *العمليات على البكسل* تحويلات تُطبق على كل بكسل بشكل مستقل:
        - *السطوع*: زيادة/تقليل شدة الإضاءة
        - *التباين*: زيادة/تقليل الفرق بين الألوان
        - *الصور السالبة*: عكس الألوان
        - *العتبة*: تحويل إلى أبيض وأسود
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            operation = st.selectbox("اختر العملية:", ("السطوع", "التباين", "صورة سالبة", "عتبة"))
            
            if operation == "السطوع":
                brightness = st.slider("مستوى السطوع", -100, 100, 0)
                result = cv2.convertScaleAbs(img_array, alpha=1, beta=brightness)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            elif operation == "التباين":
                contrast = st.slider("مستوى التباين", 0.0, 3.0, 1.0, 0.1)
                result = cv2.convertScaleAbs(img_array, alpha=contrast, beta=0)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            elif operation == "صورة سالبة":
                result = cv2.bitwise_not(img_array)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            else:
                threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                st.image(result, use_column_width=True, clamp=True)

# ==============================================================
# الوحدة 4: الفلاتر والالتفاف
# ==============================================================
elif selected == "الفلاتر والالتفاف":
    st.header("الفلاتر والالتفاف (Filtering & Convolution)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *الفلاتر والالتفاف* تستخدم نواة (Kernel) لتطبيق تأثيرات:
        - *التعزيز*: زيادة وضوح الحواف
        - *طمس*: تقليل الضوضاء وتنعيم الصورة
        - *كشف الحواف*: تحديد الحدود بين المناطق
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            filter_type = st.selectbox("اختر نوع الفلتر:", ("طمس Gaussian", "طمس Median", "تعزيز", "كشف الحواف"))
            
            if filter_type == "طمس Gaussian":
                kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                result = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            elif filter_type == "طمس Median":
                kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                result = cv2.medianBlur(img_array, kernel_size)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            elif filter_type == "تعزيز":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                result = cv2.filter2D(img_array, -1, kernel)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            else:
                kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
                result = cv2.filter2D(img_array, -1, kernel)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

# ==============================================================
# الوحدة 5: إزالة الضوضاء
# ==============================================================
elif selected == "إزالة الضوضاء":
    st.header("إزالة الضوضاء (Denoising)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *إزالة الضوضاء* تقنيات لتقليل التشوهات في الصور:
        - *ضوضاء Salt & Pepper*: نقاط سوداء وبيضاء عشوائية
        - *ضوضاء Gaussian*: تغيرات عشوائية في شدة البكسل
        - *Median Filter*: فعال لإزالة Salt & Pepper
        - *Bilateral Filter*: يحافظ على الحواف
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
            
            add_noise = st.checkbox("إضافة ضوضاء للصورة")
            if add_noise:
                noise = np.random.randint(0, 2, img_array.shape[:2], dtype=np.uint8)
                noise = noise * 255
                if len(img_array.shape) == 3:
                    noise = np.stack([noise]*3, axis=2)
                noisy_img = cv2.add(img_array, noise)
                st.image(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                img_array = noisy_img
        
        with col2:
            denoise_method = st.selectbox("اختر طريقة إزالة الضوضاء:", ("Median Filter", "Gaussian Filter", "Bilateral Filter"))
            
            if denoise_method == "Median Filter":
                kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                result = cv2.medianBlur(img_array, kernel_size)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            elif denoise_method == "Gaussian Filter":
                kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
                result = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            else:
                d = st.slider("قطر البكسل", 1, 15, 9, 2)
                result = cv2.bilateralFilter(img_array, d, 75, 75)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

# ==============================================================
# الوحدة 6: كشف الحواف
# ==============================================================
elif selected == "كشف الحواف":
    st.header("كشف الحواف (Edge Detection)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *كشف الحواف* لتحديد الحدود بين المناطق المختلفة:
        - *Sobel*: مشتقات من الدرجة الأولى
        - *Laplacian*: مشتقات من الدرجة الثانية
        - *Canny*: خوارزمية متعددة المراحل
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            edge_method = st.selectbox("اختر طريقة كشف الحواف:", ("Sobel", "Laplacian", "Canny"))
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
            if edge_method == "Sobel":
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobel_x*2 + sobel_y*2)
                sobel = np.uint8(sobel / sobel.max() * 255)
                st.image(sobel, use_column_width=True, clamp=True)
            
            elif edge_method == "Laplacian":
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian = np.uint8(np.absolute(laplacian))
                st.image(laplacian, use_column_width=True, clamp=True)
            
            else:
                threshold1 = st.slider("العتبة الدنيا", 0, 255, 100)
                threshold2 = st.slider("العتبة العليا", 0, 255, 200)
                canny = cv2.Canny(gray, threshold1, threshold2)
                st.image(canny, use_column_width=True, clamp=True)

# ==============================================================
# الوحدة 7: العمليات المورفولوجية
# ==============================================================
elif selected == "العمليات المورفولوجية":
    st.header("العمليات المورفولوجية (Morphological Operations)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *العمليات المورفولوجية* لمعالجة الصور الثنائية:
        - *التآكل (Erosion)*: يتقلص حدود الأجسام
        - *التوسيع (Dilation)*: يوسع حدود الأجسام
        - *الفتح (Opening)*: تآكل ثم توسيع
        - *الإغلاق (Closing)*: توسيع ثم تآكل
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            st.image(binary, use_column_width=True, clamp=True)
        
        with col2:
            morph_op = st.selectbox("اختر العملية المورفولوجية:", ("تآكل (Erosion)", "توسيع (Dilation)", "فتح (Opening)", "إغلاق (Closing)"))
            
            kernel_size = st.slider("حجم النواة", 3, 15, 5, 2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if morph_op == "تآكل (Erosion)":
                result = cv2.erode(binary, kernel, iterations=1)
            elif morph_op == "توسيع (Dilation)":
                result = cv2.dilate(binary, kernel, iterations=1)
            elif morph_op == "فتح (Opening)":
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            else:
                result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
# ==============================================================
# الوحدة 8: التحويلات الهندسية
# ==============================================================
elif selected == "التحويلات الهندسية":
    st.header("التحويلات الهندسية (Geometric Transforms)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *التحويلات الهندسية* تغير الشكل الهندسي للصورة:
        - *انزياح*: تحريك الصورة
        - *دوران*: تدوير الصورة
        - *قياس*: تكبير/تصغير الصورة
        - *انعكاس*: عكس الصورة
        - *اقتصاص*: قطع جزء من الصورة
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            transform = st.selectbox("اختر التحويل الهندسي:", ("انزياح", "دوران", "قياس", "انعكاس", "اقتصاص"))
            
            if transform == "انزياح":
                tx = st.slider("انزياح أفقي", -100, 100, 0)
                ty = st.slider("انزياح رأسي", -100, 100, 0)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                result = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
            
            elif transform == "دوران":
                angle = st.slider("زاوية الدوران", -180, 180, 0)
                height, width = img_array.shape[:2]
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                result = cv2.warpAffine(img_array, M, (width, height))
            
            elif transform == "قياس":
                scale = st.slider("مقياس التكبير/التصغير", 0.1, 3.0, 1.0, 0.1)
                result = cv2.resize(img_array, None, fx=scale, fy=scale)
            
            elif transform == "انعكاس":
                flip_code = st.radio("اتجاه الانعكاس:", ("أفقي", "رأسي", "كلاهما"))
                if flip_code == "أفقي":
                    result = cv2.flip(img_array, 1)
                elif flip_code == "رأسي":
                    result = cv2.flip(img_array, 0)
                else:
                    result = cv2.flip(img_array, -1)
            
            else:
                height, width = img_array.shape[:2]
                x = st.slider("بداية الأفقي", 0, width-100, 0)
                y = st.slider("بداية الرأسي", 0, height-100, 0)
                w = st.slider("العرض", 10, width-x, min(100, width-x))
                h = st.slider("الارتفاع", 10, height-y, min(100, height-y))
                result = img_array[y:y+h, x:x+w]
            
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

# ==============================================================
# الوحدة 9: المرشحات المخصصة
# ==============================================================
elif selected == "المرشحات المخصصة":
    st.header("المرشحات المخصصة والتأثيرات الفنية")
    
    with st.expander("النظرية"):
        st.markdown("""
        *المرشحات المخصصة* تتيح إنشاء تأثيرات فريدة:
        - *نواة الالتفاف*: مصفوفة تحدد كيفية دمج البكسلات
        - *تأثيرات فنية*: محاكاة الرسم، التخطيط، إلخ
        - *تحكم دقيق*: ضبط كل عنصر في النواة
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            filter_type = st.selectbox("اختر نوع المرشح:", ("تحديد الحواف", "التعزيز", "النقش", "مخصص"))
            
            if filter_type == "تحديد الحواف":
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            elif filter_type == "التعزيز":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            elif filter_type == "النقش":
                kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            else:
                st.write("حدد قيم النواة المخصصة (3x3):")
                k11 = st.slider("k11", -2.0, 2.0, 0.0, 0.1)
                k12 = st.slider("k12", -2.0, 2.0, 0.0, 0.1)
                k13 = st.slider("k13", -2.0, 2.0, 0.0, 0.1)
                k21 = st.slider("k21", -2.0, 2.0, 0.0, 0.1)
                k22 = st.slider("k22", -2.0, 2.0, 1.0, 0.1)
                k23 = st.slider("k23", -2.0, 2.0, 0.0, 0.1)
                k31 = st.slider("k31", -2.0, 2.0, 0.0, 0.1)
                k32 = st.slider("k32", -2.0, 2.0, 0.0, 0.1)
                k33 = st.slider("k33", -2.0, 2.0, 0.0, 0.1)
                kernel = np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])
            
            result = cv2.filter2D(img_array, -1, kernel)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

# ==============================================================
# الوحدة 10: تحويل المنظور
# ==============================================================
elif selected == "تحويل المنظور":
    st.header("تحويل المنظور (Perspective Transform)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *تحويل المنظور* يغير منظور الصورة:
        - *النقاط المرجعية*: 4 نقاط في الأصل و4 في الهدف
        - *مصفوفة التحويل*: تحويل بين النقاط
        - *تطبيقات*: تصحيح التشوهات، تغيير زاوية الرؤية
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = img_array.shape[:2]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            st.write("اضبط نقاط الهدف لتحويل المنظور:")
            x1 = st.slider("النقطة 1 X", 0, width, 100)
            y1 = st.slider("النقطة 1 Y", 0, height, 100)
            x2 = st.slider("النقطة 2 X", 0, width, width-100)
            y2 = st.slider("النقطة 2 Y", 0, height, 100)
            x3 = st.slider("النقطة 3 X", 0, width, width-100)
            y3 = st.slider("النقطة 3 Y", 0, height, height-100)
            x4 = st.slider("النقطة 4 X", 0, width, 100)
            y4 = st.slider("النقطة 4 Y", 0, height, height-100)
            
            src_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            dst_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(img_array, matrix, (width, height))
            
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

# # ==============================================================
# # الوحدة 11: استخراج المعالم
# # ==============================================================
# elif selected == "استخراج المعالم":
#     st.header("استخراج المعالم والوصفات (Features & Descriptors)")
    
#     with st.expander("النظرية"):
#         st.markdown("""
#         *استخراج المعالم* لتحديد النقاط المميزة:
#         - *المعالم*: زوايا، حواف، بقع مميزة
#         - *الوصفات*: متجهات رقمية تصف المنطقة
#         - *تطبيقات*: مطابقة الصور، التعرف على الأشياء
#         """)
#     st.subheader("التطبيق العملي")
#     img =
    
    
    
    
    
# ==============================================================
# الوحدة 12: تحليل الحدود
# ==============================================================
elif selected == "تحليل الحدود":
    st.header("تحليل الحدود (Contour Analysis)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *تحليل الحدود* يكتشف ويحلل حدود الأشياء:
        - *الحدود*: منحنيات تربط النقاط المتصلة
        - *الخصائص*: المساحة، المحيط، المركز
        - *تطبيقات*: عد الأشياء، التعرف على الأشكال
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
            
            # تحويل إلى تدرج رمادي وتطبيق عتبة
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            st.image(thresh, use_column_width=True, caption="بعد العتبة", clamp=True)
        
        with col2:
            # إيجاد الحدود
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # رسم الحدود
            result = img_array.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
            
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True, caption="الحدود المكتشفة")
            
            st.write(f"*عدد الحدود:* {len(contours)}")
            
            if len(contours) > 0:
                # تحليل أكبر حدود
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                st.write(f"*مساحة أكبر حدود:* {area:.2f}")
                st.write(f"*محيط أكبر حدود:* {perimeter:.2f}")

# ==============================================================
# الوحدة 13: عتبة متقدمة
# ==============================================================
elif selected == "عتبة متقدمة":
    st.header("عتبة متقدمة وتجزئة (Advanced Thresholding)")
    
    with st.expander("النظرية"):
        st.markdown("""
        *العتبة المتقدمة* تقنيات لفصل الأشياء:
        - *عتبة Otsu*: تحدد القيمة الأمثل تلقائياً
        - *عتبة تكيفية*: تتكيف مع الظروف المحلية
        - *عتبة مثلثة*: مناسبة للصور ذات الـ histogram المزدوج
        """)
    
    st.subheader("التطبيق العملي")
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        with col2:
            threshold_method = st.selectbox("اختر طريقة العتبة:", 
                                         ("عتبة بسيطة", "عتبة Otsu", "عتبة تكيفية"))
            
            if threshold_method == "عتبة بسيطة":
                threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
                _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            elif threshold_method == "عتبة Otsu":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            else:
                block_size = st.slider("حجم الكتلة", 3, 21, 11, 2)
                result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, block_size, 2)
            
            st.image(result, use_column_width=True, clamp=True)

# ==============================================================
# الوحدة 14: المعالجة بالكاميرا
# ==============================================================
elif selected == "المعالجة بالكاميرا":
    st.header("المعالجة المباشرة بالكاميرا")
    
    with st.expander("النظرية"):
        st.markdown("""
        *المعالجة المباشرة* تطبق الخوارزميات على البث الحي:
        - *معالجة كل إطار*: تطبيق العمليات على الفيديو
        - *تطبيقات*: أنظمة مراقبة، واقع معزز
        - *أداء*: معالجة في الوقت الفعلي
        """)
    
    st.subheader("التطبيق العملي")
    st.info("""
    *تعليمات الاستخدام:*
    1. اضغط على 'تشغيل الكاميرا' لبدء البث
    2. اختر العملية المطلوبة
    3. اضغط على 'إيقاف الكاميرا' عند الانتهاء
    """)
    
    # حالة الكاميرا
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("تشغيل الكاميرا" if not st.session_state.camera_active else "الكاميرا تعمل"):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("إيقاف الكاميرا" if st.session_state.camera_active else "الكاميرا متوقفة"):
            st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        operation = st.selectbox("اختر عملية المعالجة:", 
                               ("بدون معالجة", "تدرج رمادي", "كشف الحواف", 
                                "صورة سالبة", "طمس", "عتبة"))
        
        # معلمات إضافية
        if operation == "طمس":
            blur_size = st.slider("حجم الطمس", 3, 15, 5, 2)
        elif operation == "عتبة":
            threshold_value = st.slider("قيمة العتبة", 0, 255, 127)
        
        # فتح الكاميرا
        cap = cv2.VideoCapture(0)
        video_placeholder = st.empty()
        
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("فشل في قراءة الإطار")
                    break
                
                # تطبيق العملية المحددة
                if operation == "تدرج رمادي":
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                elif operation == "كشف الحواف":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                elif operation == "صورة سالبة":
                    processed_frame = cv2.bitwise_not(frame)
                elif operation == "طمس":
                    processed_frame = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
                elif operation == "عتبة":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                    processed_frame = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                else:
                    processed_frame = frame
                
                # عرض الإطار المعالج
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                      use_column_width=True, caption="البث المباشر")
                
                time.sleep(0.03)
        
        finally:
            cap.release()
            if not st.session_state.camera_active:
                video_placeholder.empty()
                st.info("تم إيقاف الكاميرا")

# ==============================================================
# الوحدة 15: المشروع الختامي
# ==============================================================
elif selected == "المشروع الختامي":
    st.header("المشروع الختامي: سلسلة عمليات معالجة الصور")
    
    st.markdown("""
    في هذه الوحدة، يمكنك تطبيق سلسلة من عمليات معالجة الصور على صورتك.
    اختر العمليات التي تريد تطبيقها بالترتيب، ثم شاهد النتيجة النهائية.
    """)
    
    img = load_image()
    if img is not None:
        img_array = np.array(img.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("الصورة الأصلية")
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("سلسلة العمليات")
            
            operations = st.multiselect(
                "اختر العمليات المطلوبة (سيتم تطبيقها بالترتيب):",
                ["تحويل إلى تدرج الرمادي", "تطبيق فلتر Blur", "كشف الحواف", 
                 "عتبة", "عمليات مورفولوجية", "تعديل التباين"],
                default=["تحويل إلى تدرج الرمادي", "تطبيق فلتر Blur", "كشف الحواف"]
            )
            
            result = img_array.copy()
            process_steps = []
            
            for op in operations:
                if op == "تحويل إلى تدرج الرمادي":
                    if len(result.shape) == 3:
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    process_steps.append("تحويل إلى تدرج الرمادي")
                
                elif op == "تطبيق فلتر Blur":
                    blur_size = st.slider("حجم فلتر Blur", 3, 15, 5, 2)
                    if len(result.shape) == 2:
                        result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)
                    else:
                        result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)
                    process_steps.append(f"تطبيق فلتر Blur بحجم {blur_size}")
                
                elif op == "كشف الحواف":
                    if len(result.shape) == 3:
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    result = cv2.Canny(result, 100, 200)
                    process_steps.append("كشف الحواف")
                
                elif op == "عتبة":
                    if len(result.shape) == 3:
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
                    process_steps.append("عتبة ثنائية")
                
                elif op == "عمليات مورفولوجية":
                    if len(result.shape) == 3:
                        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    kernel_size = st.slider("حجم النواة المورفولوجية", 3, 15, 5, 2)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                    process_steps.append(f"عمليات مورفولوجية بحجم نواة {kernel_size}")
                
                elif op == "تعديل التباين":
                    if len(result.shape) == 2:
                        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                    contrast = st.slider("مستوى التباين", 0.0, 3.0, 1.5, 0.1)
                    result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
                    process_steps.append(f"تعديل التباين بمستوى {contrast}")
            
            # عرض النتيجة
            st.subheader("النتيجة النهائية")
            if len(result.shape) == 2:
                st.image(result, use_column_width=True, clamp=True)
            else:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # عرض خطوات المعالجة
            st.subheader("خطوات المعالجة المطبقة")
            for i, step in enumerate(process_steps, 1):
                st.write(f"{i}. {step}")
            
            # خيار حفظ الصورة
            if st.button("حفظ الصورة الناتجة"):
                if len(result.shape) == 2:
                    result_pil = Image.fromarray(result)
                else:
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)
                
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                href = f'<a href="data:file/png;base64,{base64.b64encode(byte_im).decode()}" download="processed_image.png">اضغط هنا لتحميل الصورة</a>'
                st.markdown(href, unsafe_allow_html=True)

# ==============================================================
# رسالة إذا لم يتم اختيار أي وحدة
# ==============================================================
else:
    st.info("مرحباً! يرجى اختيار وحدة من الشريط الجانبي لبدء معالجة الصور.")

# معلومات التطبيق في الشريط الجانبي
with st.sidebar:
    st.markdown("---")
    st.markdown("### معلومات التطبيق")
    st.markdown("""
    *معالجة الصور التفاعلية*
    
    تطبيق تعليمي شامل لمعالجة الصور
    باستخدام Streamlit وOpenCV
    
    *المكتبات المستخدمة:*
    - Streamlit
    - OpenCV
    - NumPy
    - PIL
    """)