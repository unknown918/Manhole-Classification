import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

st.title('井盖数据分析')

df = pd.DataFrame(np.random.randn(50, 2) / [50, 50] + [28.17022, 112.92821],
                  columns=['lat', 'lon'])

col1, col2 = st.columns(2)
col1.subheader('空间分布')
col1.map(df)

col2.subheader('种类统计')

# 类别
categories = ['good', 'broke', 'lose', 'uncovered', 'circle']

# 每个类别的数量
counts = [20, 7, 9, 11, 3]

# 创建一个DataFrame
df = pd.DataFrame({'类别': categories, '数量': counts})

# 在Streamlit应用中创建柱状图
col2.bar_chart(df.set_index('类别'))
col11,col22,col3,col4,col5 = st.columns(5)
col11.metric('good', '20', '4')
col22.metric('broke', '7', '3')
col3.metric('lose', '9', '0')
col4.metric('uncovered', '11', '5')
col5.metric('circle', '3', '1')