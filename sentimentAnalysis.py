import warnings

# Chỉ ignore đúng warning của Keras về cấu trúc inputs
warnings.filterwarnings(
    "ignore",
    message="The structure of `inputs` doesn't match the expected structure.",
    category=UserWarning,
)
import pandas as pd
import streamlit as st
import numpy as np
import torch
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModel, AutoTokenizer

from streamlit_option_menu import option_menu
from cleandata import Text_PreProcessing_util
# Import thêm các thư viện cần thiết cho code chuyên đề
import tensorflow as tf
from pyvi import ViTokenizer
import pickle


# Load model và tokenizer cho chuyên đề

def load_model_and_tokenizer():
    model = load_model('model/model_bilstm_cnn.keras')
    with open("model/tokenizer_data (100).pkl", "rb") as input_file:
        tokenizer = pickle.load(input_file)
    return model, tokenizer

# Load model và tokenizer
my_model, my_tokenizer = load_model_and_tokenizer()

# Các hàm cho chuyên đề
def preprocess_raw_input(raw_input, tokenizer):
    input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(raw_input))
    input_text_pre = " ".join(input_text_pre)
    input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=100)
    return vec_data

def inference_model(input_feature, model):
    output = model(input_feature).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {'Tiêu cực': 0, 'Tích cực': 1, 'Trung lập': 2}
    label = list(label_dict.keys())
    return label[int(result)], conf

def prediction(raw_input, tokenizer, model):
    input_model = preprocess_raw_input(raw_input, tokenizer)
    result, conf = inference_model(input_model, model)
    return result, conf


# Giao diện Streamlit
st.set_page_config(
    page_title="Xử lý ngôn ngữ tự nhiên",
    page_icon="😊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: red;
        text-align: center;
    }

    </style>
    <div class="footer">
        <p>Website được tạo bởi Lê Cẩm Bằng</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Hàm để hiển thị ứng dụng Sentiment Analysis
def sentiment_analysis_app():
    st.title("Phân tích cảm xúc của người mua hàng 😊🤔😢")
    # Add images using HTML and CSS
    st.markdown("""
        <style>
            .image {
                width: 24px;
                height: 24px;
                margin-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    decor = Image.open("decoration.png")
    st.image(decor)

    # Nhập văn bản đầu vào
    user_input = st.text_input("Nhập văn bản", placeholder="Nhập văn bản tại đây...")
    if st.button("Phân tích"):
        if user_input:
            try:
                # Tiền xử lý văn bản đầu vào
                processed_input = Text_PreProcessing_util([user_input])
                # Dự đoán cảm xúc - SỬ DỤNG CODE CHUYÊN ĐỀ
                result, conf = prediction(processed_input[0], my_tokenizer, my_model)
                if result == "Tích cực":
                    st.success(f"Dự đoán cảm xúc: **Tích cực**")
                elif result == "Tiêu cực":
                    st.error(f"Dự đoán cảm xúc: **Tiêu cực**")
                else:
                    st.info(f"Dự đoán cảm xúc: **Trung lập**")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Vui lòng nhập văn bản!")


# Hàm để hiển thị ứng dụng Recommendation
def recommendation_app():
    st.title("Khuyến nghị Mua hàng Dựa trên đánh giá Cảm xúc")


    # File uploader
    uploaded_file = st.file_uploader("Duyệt tập dữ liệu", type=["csv", "txt"], label_visibility="visible")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.write(f"Tệp đã được tải lên: {file_name}")

        sentences=[]
        # Read file into a pandas dataframe if it's a CSV
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.write("Các hàng đầu tiên của tệp:")
            st.write(df.head())  # Display first few rows of the uploaded CSV

            # Get list of sentences (assuming one column of text)
            if 'text' in df.columns:
                sentences = df['text'].tolist()
            else:
                st.error("Không tìm thấy cột 'text' trong tệp CSV.")

        # Read content if it's a TXT file
        elif file_name.endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")
            sentences = content.split("\n")
            st.text(content)  # Display content of the uploaded text file
        else:
            st.error("Loại tệp không được hỗ trợ. Vui lòng tải lên tệp CSV hoặc TXT.")

        if 'sentences' in locals() and sentences:
            st.write("Các câu trong tệp:")

            with st.spinner("Đang xử lý... Vui lòng chờ trong khi chúng tôi phân tích các câu"):
                results = []

                positive_count = 0
                negative_count = 0
                neutral_count = 0

                for i, sentence in enumerate(sentences):
                    # Tiền xử lý văn bản
                    processed_input = Text_PreProcessing_util([sentence])[0]
                    # Dự đoán cảm xúc - SỬ DỤNG CODE CHUYÊN ĐỀ
                    result, conf = prediction(processed_input, my_tokenizer, my_model)
                    results.append({"Câu": sentence, "Cảm xúc": result})

                    if result == "Tích cực":
                        positive_count += 1
                    elif result == "Tiêu cực":
                        negative_count += 1
                    else:
                        neutral_count += 1

                # Chuyển danh sách kết quả thành DataFrame
                results_df = pd.DataFrame(results)

                # Tính tỷ lệ phần trăm
                total_count = len(sentences)
                positive_percentage = (positive_count / total_count) * 100
                negative_percentage = (negative_count / total_count) * 100
                neutral_percentage = (neutral_count / total_count) * 100

                # Hiển thị kết quả
                st.write("Kết quả:")
                st.dataframe(results_df)
                st.success("Đã hoàn tất xử lý tất cả câu!")

                # Hiển thị tỷ lệ phần trăm cảm xúc tích cực và tiêu cực
                st.write(f"Cảm xúc Tích cực: {positive_percentage:.2f}%")
                st.write(f"Cảm xúc Tiêu cực: {negative_percentage:.2f}%")
                st.write(f"Cảm xúc Trung lập: {neutral_percentage:.2f}%")

                 # Tạo dictionary để so sánh tỷ lệ phần trăm
                sentiment_percentages = {
                    "Tích cực": positive_percentage,
                    "Tiêu cực": negative_percentage,
                    "Trung lập": neutral_percentage
                 }

                # Tìm cảm xúc có tỷ lệ phần trăm cao nhất
                max_sentiment = max(sentiment_percentages, key=sentiment_percentages.get)
                max_percentage = sentiment_percentages[max_sentiment]


                # Quyết định có nên mua hay không
                if max_sentiment == "Tích cực":
                    st.success(f"Khuyến nghị: **Nên mua**, bạn nên mua sản phẩm này.")
                elif max_sentiment == "Tiêu cực":
                    st.error(f"Khuyến nghị: **Không nên mua**, bạn không nên mua sản phẩm này.")
                else:
                    st.info(f"Đề xuất: **Hãy cân nhắc**, bạn nên cân nhắc trước khi mua sản phẩm này.")

        else:
            st.warning("Không tìm thấy câu trong tệp đã tải lên.")


selected = option_menu(
    menu_title=None, #required
    options=["Trang chủ", "Phân tích cảm xúc", "Khuyến nghị Mua hàng"], #required
    icons=["house", "emoji-smile", "archive"], #optional
    menu_icon="cast", #optional
    default_index=0, #optional
    orientation="horizontal",
    styles={
        "container": {"padding": "1!important"},
        "icon": {"color": "orange", "font-size": "20px"},
        "nav-link": {
            "font-size": "20px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
            "font-weight": "normal"
        },
        "nav-link-selected": {"background-color": "green","font-weight": "normal"},
    },
)
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)
def home():
    st.title("Xử lý ngôn ngữ tự nhiên")
    st.title("Xây dựng mô hình phân tích cảm tính của khách hàng trên nền tảng thương mại điện tử Shopee")

if selected == "Phân tích cảm xúc":
    sentiment_analysis_app()
elif selected == "Khuyến nghị Mua hàng":
    recommendation_app()
elif selected == "Trang chủ":
    home()