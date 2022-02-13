import os
import random
import shutil
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
import umap.plot
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from sklearn.metrics import silhouette_score, classification_report, plot_confusion_matrix
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
random.seed(0)
tf.random.set_seed(0)
np.random.seed(0)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

np.set_printoptions(threshold=sys.maxsize)
matplotlib.use('agg')


def manifold_learning(hl, n_dim, n_neighbors=10, min_dist=0, metric="euclidean", label=None):
    # find manifold on autoencoded embedding
    mapper = umap.UMAP(random_state=42, metric=metric, n_components=n_dim,
                       n_neighbors=n_neighbors, min_dist=min_dist).fit(hl, y=label)
    hle = mapper.transform(hl)

    return hle, mapper

def clustering(hle, k, cluster="GMM"):
    y_pred = None
    model = None
    # clustering on new manifold of autoencoded embedding
    if cluster == 'GMM':
        model = GaussianMixture(covariance_type='full', n_components=k, random_state=42)
        model.fit(hle)
        y_pred_prob = model.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
        print(model.converged_)
        print(model.weights_)
    elif cluster == "BGMM":
        model = BayesianGaussianMixture(covariance_type='full', n_components=30, random_state=42)
        model.fit(hle)
        y_pred_prob = model.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
        print(model.converged_)
        print(model.weights_)
    elif cluster == 'KM':
        model = KMeans(init='k-means++', n_clusters=k, random_state=42, n_init=20)
        y_pred = model.fit_predict(hle)
    elif cluster == 'SC':
        model = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
        y_pred = model.fit_predict(hle)

    print(f"silhouette_score : {silhouette_score(hle, y_pred)}")

    return model, y_pred

@nb.jit
def select_optimal_k(hle, min, max):
    BIC = []
    AIC = []
    Silhouette = []
    for k in range(min, max + 1):
        if k >= len(hle):
            max = k - 1
            break
        gmm = GaussianMixture(covariance_type='full', n_components=k, random_state=42)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
        BIC.append(gmm.bic(hle))
        AIC.append(gmm.aic(hle))
        Silhouette.append(silhouette_score(hle, y_pred))
    if len(BIC) == 0:
        return None
    plt.figure(figsize=(17, 10))
    ax1, ax2 = plt.gca(), plt.gca().twinx()
    ax1.plot(range(min, max + 1), BIC, "ro-", label="BIC")
    ax1.plot(range(min, max + 1), AIC, "bs--", label="AIC")
    ax1.set_xlabel("K", fontsize=15)
    ax1.set_ylabel("Information criterion", fontsize=17, labelpad=12)
    ax1.legend(loc="upper left", ncol=2)
    ax1.annotate("Opitimal K", xy=(np.argmin(BIC) + min, np.min(BIC)),
                 xytext=(np.argmin(BIC) + min - 2, np.min(BIC) + 1000),
                 fontsize=15, arrowprops={"facecolor": 'black', "shrink": 0.13, "width": 2})
    ax2.plot(range(min, max + 1), Silhouette, "gv-.", label="silhouette score")
    ax2.set_ylabel("Silhouette score", fontsize=17, labelpad=12)
    ax2.legend(loc="upper right")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Select Optimal K", fontsize=20)
    plt.savefig(f"C:/Users/user/Desktop/마약지문/optimal_k({np.argmin(BIC) + min}).png")

    return np.argmin(BIC) + min

def semi_supervised_learning(hle, label, kernel, n):
    model = LabelPropagation(kernel=kernel, n_neighbors=n)
    # model = LabelSpreading(kernel=kernel, n_neighbors=n)
    model.fit(hle, label)

    return model

def visualization(hle, df):
    # cluster_features, _ = manifold_learning(hle, 3)
    df.loc[df["y-hat"] == 2, "cluster"] += 20
    df.loc[df["y-hat"] == 3, "cluster"] += 24
    # fig = px.scatter_3d(x=cluster_features[:, 0], y=cluster_features[:, 1], z=cluster_features[:, 2], color=df["cluster"])

    cluster_features_2d, mapper = manifold_learning(hle, 2)
    fig = px.scatter(x=cluster_features_2d[:, 0], y=cluster_features_2d[:, 1], color=df["cluster"])
    fig.show()
    fig, ax = plt.subplots(figsize=(15, 10))
    umap.plot.points(mapper, labels=df["cluster"], ax=ax)
    plt.savefig("C:/Users/user/Desktop/마약지문/clustering_result/결과/이미지모음/최종/2d_scatter.png")


x = np.load("./Data/합성법_internal_scaling_50%_dataset_306(800x400).npy")
x = x / 255.
hog = np.load("./Data/합성법_internal_scaling_50%_hog_dataset_306(800x400).npy")

method1 = "합성법"

# model load
encoder1 = tf.keras.models.load_model(
    f"C:/Users/user/Desktop/마약지문/models/{method1}_100_16_binary_crossentropy_encoder.h5")

# labeled data
labeled_df = pd.read_excel("C:/Users/user/Desktop/이온정리2.xlsx", sheet_name="sample(18개) 정보")
labeled_y = labeled_df["type"]

# whole data
df = pd.read_excel("C:/Users/user/Desktop/dataset.xlsx")
df = pd.merge(df, labeled_df.iloc[:, [1, 2]], left_on="file_path", right_on="file_name", how="left")
df.drop(columns=["Unnamed: 0", "file_name"], inplace=True)
df.fillna(-1, inplace=True)
y = df["type"]
print(df)

# ---------------------------------------- Extracted image feature ----------------------------------------
hl = encoder1.predict(x)
hle, _ = manifold_learning(hl, 100, label=y)
hog, _ = manifold_learning(hog, 100, metric="cosine", label=y)
concat_feature = np.concatenate([hle, hog], axis=1)
features, _ = manifold_learning(concat_feature, 30, label=y)

df.drop(columns=["f1", "f2", "f3"], inplace=True)
df = pd.concat([df, pd.DataFrame(features)], axis=1)
print(df)

print(
    f"---------------------------------------- Feature shape: {features.shape} ----------------------------------------")

labeled_df = df.loc[df.file_path.isin(labeled_df.file_name), :]

# 세가지 합성법 샘플 (201개), train
selected_df = df.loc[df.Method.isin(["Iα", "Iβ", "II", "III"]), :]
selected_df.replace({"Iα": 1, "Iβ": 1, "II": 2, "III": 3}, inplace=True)
print(selected_df.shape)

# IV, V, VI에 해당되는 샘플(105개)
val_df = df.loc[~df.Method.isin(["Iα", "Iβ", "II", "III"]), :]

print(
    f"---------------------------------------- 1st discrimination feature shape: {selected_df.shape} ----------------------------------------")

# unlabeled sample
validation_df = selected_df.loc[~selected_df.file_path.isin(labeled_df.file_path), :]

# ---------------------------------------- semi supervised learning ----------------------------------------
print("---------------------------------------- 1st discrimination ----------------------------------------")

# semi supervised learning result
model = semi_supervised_learning(features[selected_df.index], y[selected_df.index], "knn", 3)
y_hat = model.transduction_
df.loc[selected_df.index, "y-hat"] = y_hat
selected_df.loc[:, "y-hat"] = y_hat
print(selected_df)

# 세 가지 제조법에 해당하는 샘플에 대한 semi-supervised learning의 결과 (labeled data 제외)
print(classification_report(y_pred=selected_df.loc[validation_df.index, "y-hat"], y_true=validation_df["Method"],
                            target_names=["I", "II", "III"],
                            digits=3))
# plot_confusion_matrix(model, features[validation_df.index], validation_df["Method"], cmap=plt.cm.hot)
# plt.savefig("C:/Users/user/Desktop/semi-supervised learning(100-100-30, knn 3, umap_neighbor 10).png")

# IV, V, VI에 해당되는 샘플(105개)에 대한 합성법 판별
y_hat2 = model.predict(features[val_df.index])
df.loc[val_df.index, "y-hat"] = y_hat2
val_df.loc[:, "y-hat"] = y_hat2

method2 = "Ions"
encoder2 = tf.keras.models.load_model(
    f"C:/Users/user/Desktop/models/{method2}_100_16_binary_crossentropy_encoder.h5")

x2 = np.load("./Data/ions_internal_scaling_dataset_306(800x400).npy")
x2 = x2 / 255.
hog2 = np.load("./Data/ions_internal_scaling_hog_dataset_306(800x400).npy")
print(hog2.shape)

hl2 = encoder2.predict(x2)
hle2, _ = manifold_learning(hl2, 100)
hog2, _ = manifold_learning(hog2, 100)
concat_feature = np.concatenate([hle2, hog2], axis=1)
cluster_features, _ = manifold_learning(concat_feature, 30)
print(
    f"---------------------------------------- Feature shape: {cluster_features.shape} ----------------------------------------")

# ---------------------------------------- clustering step ----------------------------------------
print("---------------------------------------- 2nd discrimination ----------------------------------------")

cluster_model = []
K = []
df.loc[:, "train_val"] = ""
for label in sorted(selected_df["y-hat"].unique()):
    # 1차 기준(합성법)으로 분류한 데이터를 대상으로 클러스터링
    Method_df = selected_df.loc[selected_df["y-hat"] == label, :]
    optimal_k = select_optimal_k(cluster_features[Method_df.index], min=2, max=25)
    K.append(optimal_k)
    print(optimal_k)
    if optimal_k is None:
        cluster_pred = 0
        selected_df.loc[Method_df.index, "cluster"] = cluster_pred
        df.loc[Method_df.index, "cluster"] = cluster_pred
        continue
    if label == 1:
        optimal_k = 20
    if label == 2:
        optimal_k = 4
    cluster, cluster_pred = clustering(cluster_features[Method_df.index], optimal_k)
    cluster_model.append(cluster)
    selected_df.loc[Method_df.index, "cluster"] = cluster_pred
    df.loc[Method_df.index, "cluster"] = cluster_pred
    df.loc[Method_df.index, "train_val"] = "Train"

for idx, (label, cluster) in enumerate(zip(sorted(val_df["y-hat"].unique()), cluster_model)):
    # 1차 기준(합성법)으로 분류한 데이터를 대상으로 클러스터링
    Method_df = val_df.loc[val_df["y-hat"] == label, :]
    cluster_pred = cluster.predict(cluster_features[Method_df.index])
    val_df.loc[Method_df.index, "cluster"] = cluster_pred
    df.loc[Method_df.index, "cluster"] = cluster_pred
    df.loc[Method_df.index, "train_val"] = "Val"

# df.to_excel("C:/Users/user/Desktop/clustering_result/semi_supervised learning+clustering.xlsx")
visualization(concat_feature, df)

# ---------------------------------------- Save image ----------------------------------------

image_dir = "C:/Users/user/Desktop/ions(32 ions)/internal scaling(306)"

file_path = []
for file in os.listdir(image_dir):
    abs_path = os.path.join(image_dir, file)
    file_path.append(abs_path)
df = [file_path, df["y-hat"], df.cluster, df.train_val]
df = pd.DataFrame(df, index=["file_path", "y-hat", "cluster", "train_val"])
df = df.T

print(df)

method = "Semi-supervised laerning + GMM"
for y_hat in df["y-hat"].unique():
    dir = f"C:/Users/user/Desktop/clustering_result/"
    if not os.path.exists(dir):
        os.mkdir(dir)

    if not os.path.exists(os.path.join(dir, "N2D_" + method + f"_합성법 {int(y_hat)}_24")):
        method_dir = os.path.join(dir, "N2D_" + method + f"_합성법 {int(y_hat)}_24")
        os.mkdir(method_dir)
    else:
        method_dir = os.path.join(dir, "N2D_" + method + f"_합성법 {int(y_hat)}_24")

    for cluster_index in df.loc[df["y-hat"] == y_hat, "cluster"].unique():
        if not os.path.exists(os.path.join(method_dir, f"cluster - {str(cluster_index)}")):
            cluster_dir = os.path.join(method_dir, f"cluster - {str(cluster_index)}")
            os.mkdir(cluster_dir)
        else:
            cluster_dir = os.path.join(method_dir, f"cluster - {str(cluster_index)}")

        file_path = df.loc[(df["y-hat"] == y_hat) & (df.cluster == cluster_index), "file_path"].tolist()
        train_val = df.loc[(df["y-hat"] == y_hat) & (df.cluster == cluster_index), "train_val"].tolist()
        for each_file_path, tv in zip(file_path, train_val):
            shutil.copy(each_file_path, os.path.join(cluster_dir, each_file_path.split("\\")[-1]))
            shutil.move(os.path.join(cluster_dir, each_file_path.split("\\")[-1]),
                        os.path.join(cluster_dir, (each_file_path.split("\\")[-1].split(".")[0] + f"_{tv}.png")))

