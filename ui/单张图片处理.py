import streamlit as st

from PIL import Image

# 两列 宽度1:2
col3, col4 = st.columns([1, 2])
# 第一列 logo图片 100px宽度 高度自适应
col3.image(Image.open(r'ui/logo.png'), width=150)
# 第二列 标题
col4.title('智能井盖监测系统')

col5, col6 = (st.columns(2))
file=col5.file_uploader('选择井盖图片上传')
col6.text_input('位置')
col6.text_input('上传者')

# file即为上传图片


# 处理图片
pass
# result改成file打框后的结果
result = file

# 将file展示出来
if file is not None:
# 两列
    col1, col2 = st.columns(2)
    # 第一列展示上传的图片
    col1.image(file, caption='Uploaded Image.', use_column_width=True)
    # 第二列展示处理后的图片
    col2.image(result, caption='Processed Image.', use_column_width=True)
