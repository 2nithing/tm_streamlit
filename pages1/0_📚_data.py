import streamlit as st
st.set_page_config(
        page_title='data',
        page_icon="📚",
    )
st.header('Upload data')
st.logo('ultra_final.png')
col1,col2 = st.columns([1,1])
cont1 = col1.container(border=True)
class1 = cont1.text_input("Class name",key='class1')
class_title1 = class1 if class1 else ""
images1 = cont1.file_uploader(f"Upload {class_title1} images",accept_multiple_files=True,type=['png','jpg'],key='files1')
cont2 = col2.container(border=True)
class2 = cont2.text_input("Class name",key='class2')
class_title2 = class2 if class2 else ""
images2 = cont2.file_uploader(f"Upload {class_title2} images",accept_multiple_files=True,type=['png','jpg'],key='files2')
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}

</style>""", unsafe_allow_html=True)

tab1,tab2 = st.tabs(["data","Train"])