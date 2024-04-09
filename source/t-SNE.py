import os

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, TapTool, CDSView, GroupFilter
from bokeh.plotting import figure, output_file, save
from keras.models import load_model
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 加载 Keras 模型
model = load_model('models/res_mobile.keras')


# 提取图像特征
def extract_features(img_path, model):
    # 加载图片并转换为模型可接受的格式
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 使用模型预测图像特征
    features = model.predict(x)
    return features.flatten()


# 从文件夹加载图像路径
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            images.append(img_path)
    return images


# 加载所有子文件夹的图像路径
root_folder = 'train'
subfolders = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if
              os.path.isdir(os.path.join(root_folder, f))]

all_image_files = []
all_image_labels = []
for folder in subfolders:
    image_files = load_images_from_folder(folder)
    all_image_files.extend(image_files)
    all_image_labels.extend([folder.split('/')[-1]] * len(image_files))  # 使用文件夹名称作为标签

# 提取所有图像的特征向量
features_list = [extract_features(img_path, model) for img_path in all_image_files]
X = np.array(features_list)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

# 创建 ColumnDataSource
source = ColumnDataSource(data=dict(x=X_tsne[:, 0], y=X_tsne[:, 1], labels=all_image_labels, images=all_image_files))

# 定义标签和颜色映射
unique_labels = list(set(all_image_labels))
label_colors = {label: color for label, color in zip(unique_labels, ['blue', 'green', 'orange', 'red', 'purple'])}

# 创建 Bokeh 图形
p = figure(tools="tap", width=800, height=600, title="t-SNE Cluster Plot")
for label, color in label_colors.items():
    view = CDSView(source=source, filters=[GroupFilter(column_name='labels', group=label)])
    p.scatter(x='x', y='y', size=10, fill_color=color, legend_label=label, source=source, view=view)

# 添加点击回调
callback = CustomJS(args=dict(source=source), code="""
    var selected_index = cb_data.source.selected.indices[0];
    var url = cb_data.source.data['images'][selected_index];
    var new_page = window.open();
    new_page.document.open();
    new_page.document.write('<html><body><img src="' + url + '"></body></html>');
    new_page.document.close();
""")
p.add_tools(TapTool(callback=callback))

# 指定输出 HTML 文件
output_file("t-SNE.html")

# 保存图形为 HTML 文件
save(column(p))
