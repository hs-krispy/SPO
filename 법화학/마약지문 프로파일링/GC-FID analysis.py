import os
import shutil
import sys
import datetime as dt
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import cv2
import scipy.signal as ss
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

class FID:
    def __init__(self, data, file_names, Standardization=True, main_dir_path="C:/Users/user/Desktop/FID 이미지 분석",
                 figure_dir_name="Figures",
                 img_dir_path="C:/Users/user/Desktop/마약지문감정 데이터/processed_FID", result_dir_name="Similarity_result"):
        if Standardization:
            timeseries = np.transpose(np.array(data))
            scaler = StandardScaler()
            timeseries = scaler.fit_transform(timeseries)
            timeseries = np.transpose(timeseries)
            self.data = timeseries
        else:
            self.data = np.array(data)

        self.file_names = file_names

        # 메인 디렉토리를 지정 또는 지정 후 생성 (원하는 곳으로 경로 변경 가능)
        self.main_dir = main_dir_path
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)

        # 분석 과정에서 생성되는 figure를 저장할 디렉토리
        self.figure_dir = os.path.join(self.main_dir, figure_dir_name)
        if not os.path.exists(self.figure_dir):
            os.mkdir(self.figure_dir)

        # 이미지가 존재하는 디렉토리
        self.img_dir = img_dir_path

        # 유사도 결과를 저장할 디렉토리
        self.result_dir = os.path.join(self.main_dir, result_dir_name)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.cor_df = None

    def save_warping_paths(self, seq_1, seq_2, length=120, constraint_step=12, file_name="Dtw_warping_path.png"):
        d, paths = dtw.warping_paths(seq_1[:length], seq_2[:length], window=constraint_step)
        best_path = dtw.best_path(paths)
        dtwvis.plot_warpingpaths(seq_1[:length], seq_2[:length], paths, best_path)
        save_path = os.path.join(self.figure_dir, file_name)
        plt.savefig(save_path, dpi=400)

    # aligned data 시각화
    def save_warping(self, sample_data, constraint_step=12):
        plt.figure(figsize=(15, 8))
        cmap = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(sample_data))]
        for data, color in zip(sample_data, cmap):
            plt.plot(range(1200), data, color=color, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "Before_align.png"), dpi=400)
        plt.cla()

        plt.figure(figsize=(15, 8))
        for data in sample_data:
            d, paths = dtw.warping_paths(data, sample_data[0], window=constraint_step)
            best_path = dtw.best_path(paths)
            aligned_data = dtw.warp(data, sample_data[0], path=best_path)[0]
            plt.plot(range(1200), aligned_data, color=color, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "After_align.png"), dpi=400)

    # Make sub-sequence matrix
    def Sliding_window(self, stride=1, window_size=120):
        print(
            f"\n------------------------------- Sliding window -------------------------------")
        subsequence_matrix = []
        ts = self.data.copy()

        for idx in range(ts.shape[0]):
            prev = 0
            subsequence = []
            while prev + window_size <= ts.shape[1]:
                subsequence.append(ts[idx, prev:prev + window_size])
                prev += stride
            subsequence_matrix.append(subsequence)
        print(
            f"\nSub-sequence matrix shape : {np.array(subsequence_matrix).shape}")

        return np.array(subsequence_matrix)

    # Cluster-based Similarity Partitioning Algorithm
    def CSPA(self, data, n_clusters, constraint_step=12):
        partition = data.shape[1]
        hypergraph = np.zeros((data.shape[0], n_clusters * partition))
        sub_cluster = 0

        print(
            f"\n------------------------------- Learning CSPA -------------------------------")

        for idx in tqdm(range(partition)):
            tk = TimeSeriesKMeans(n_clusters=n_clusters, n_init=1, random_state=42, metric="dtw", init="k-means++",
                                  metric_params={"global_constraint": "sakoe_chiba",
                                                 "sakoe_chiba_radius": constraint_step,
                                                 "n_jobs": -1}, n_jobs=-1)
            res = tk.fit_predict(data[:, idx, :])
            for label in range(n_clusters):
                hypergraph[np.where(res == label)[0], sub_cluster * n_clusters + label] += 1
            sub_cluster += 1

        similarity_matrix = np.dot(hypergraph, hypergraph.T) / partition

        return similarity_matrix

    # plot heatmap & dendrogram
    def Correlation_heat_dendrogram(self, data, figure_title="CSPA result", show_figrue=True):
        similarity_matrix = data
        # Initialize figure by creating upper dendrogram
        fig = ff.create_dendrogram(similarity_matrix, labels=self.file_names, orientation="bottom", color_threshold=1.9,
                                   linkagefun=lambda x: linkage(x, "ward", metric="correlation"))

        for i in range(len(fig['data'])):
            fig['data'][i]['yaxis'] = 'y2'

        # Create Side Dendrogram
        dendro_side = ff.create_dendrogram(similarity_matrix, orientation="right", color_threshold=1.9,
                                           linkagefun=lambda x: linkage(x, "ward", metric="correlation"))

        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'

        # Add Side Dendrogram Data to Figure
        for data in dendro_side['data']:
            fig.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))
        dist_matrix = pdist(similarity_matrix, metric="correlation")
        heat_data = squareform(dist_matrix)
        # heat_data = 1 / (1 + heat_data)
        heat_data = 1 - heat_data
        similarity_matrix = np.round(heat_data, 4)

        heat_data = heat_data[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        hovertext = list()
        x, y, z = fig['layout']['xaxis']['ticktext'], fig['layout']['xaxis']['ticktext'], heat_data
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(
                    f'File X: {xx}<br />File Y: {yy}<br />Correlation similarity: {round(z[yi][xi], 5)}')

        heatmap = go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Blues',
            hoverinfo="text",
            text=hovertext
        )

        heatmap['x'] = fig['layout']['xaxis']['tickvals']
        heatmap['y'] = dendro_side['layout']['yaxis']['tickvals']

        # Add Heatmap Data to Figure
        fig.add_trace(heatmap)

        # Edit xaxis
        fig.update_layout(xaxis={'domain': [.15, 1],
                                 'mirror': False,
                                 'showgrid': False,
                                 'showline': False,
                                 'zeroline': False,
                                 'ticks': ""})
        # Edit xaxis2
        fig.update_layout(xaxis2={'domain': [0, .15],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'showticklabels': False,
                                  'ticks': ""})
        # Edit yaxis
        fig.update_layout(yaxis={'domain': [0, .85],
                                 'mirror': False,
                                 'showgrid': False,
                                 'showline': False,
                                 'zeroline': False,
                                 'showticklabels': False,
                                 'ticks': ""})
        # Edit yaxis2
        fig.update_layout(yaxis2={'domain': [.825, .975],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'showticklabels': False,
                                  'ticks': ""})

        fig.update_layout(title={
            "text": f"{figure_title}",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 0.95,
            "font": {"size": 25}
        }, width=1200, height=1200, hovermode='closest')
        fig.update_xaxes(
            tickangle=60
        )

        # Correlation dataframe
        self.cor_df = pd.DataFrame(similarity_matrix, index=self.file_names, columns=self.file_names)

        if show_figrue:
            fig.show()

    def Saving_similar_image(self, time_lag=12, correlation_threshold=0.9, NCC_threshold=0.95):
        ts = self.data.copy()

        def ccf(x, y, lag_max=1):
            result = ss.correlate(y - np.mean(y), x - np.mean(x)) / (np.std(y) * np.std(x) * len(y))
            length = (len(result) - 1) // 2
            lo = length - lag_max
            hi = length + (lag_max + 1)

            return max(result[lo:hi])

        # Normalized Cross-correlation dataframe
        ccor_matrix = pdist(ts, metric=lambda u, v: ccf(u, v, time_lag))
        ccor_arr = squareform(ccor_matrix)
        ccor_df = pd.DataFrame(ccor_arr, index=self.file_names, columns=self.file_names)
        for i in range(len(self.file_names)):
            ccor_df.iloc[i, i] = 1

        print(
            f"\n------------------------------- Save similar image -------------------------------")

        now = dt.datetime.now()
        when = now.strftime(f"%Y_%m_%d_%H_%M_R({correlation_threshold})_NCC({NCC_threshold})")
        each_dir = os.path.join(self.result_dir, when)
        os.mkdir(each_dir)

        count = 0
        total_sim_matrix = []
        for idx, file in enumerate(tqdm(self.file_names)):
            cor_idx = np.where(self.cor_df.iloc[idx, :].values >= correlation_threshold)[0]
            ccor_idx = np.where(ccor_df.iloc[idx, :].values >= NCC_threshold)[0]

            # 두 조건에 모두 해당되는 데이터 index
            selected_idx = list(set(cor_idx) & set(ccor_idx))
            predict_sim = self.cor_df.iloc[idx, selected_idx]
            # 자기 자신만 해당되는 경우는 제외
            if len(predict_sim) == 1:
                continue
            # similarity table 생성 (내림차순)
            sim_table = predict_sim.loc[~predict_sim.index.isin([file])]
            M_index = pd.MultiIndex.from_arrays([[file] * len(sim_table), sim_table.index], names=["Target", "Similar"])
            each_sim_df = pd.DataFrame(sim_table.values, index=M_index, columns=["Correlation coefficient"])
            total_sim_matrix.append(each_sim_df)

            count += 1
            if not os.path.exists(os.path.join(each_dir, file)):
                file_dir = os.path.join(each_dir, file)
                os.mkdir(file_dir)
            else:
                file_dir = os.path.join(each_dir, file)

            for target, sim in zip(predict_sim.index, predict_sim.values):
                each_file_path = os.path.join(self.img_dir, (str(target) + ".png"))
                shutil.copy(each_file_path, os.path.join(file_dir, (str(target) + ".png")))
                if file == target:
                    continue
                old = os.path.join(file_dir, (str(target) + ".png"))
                os.rename(old, old[:-4] + f" ({sim}).png")
        print(f"\n{count} samples have similar other samples")

        pd.concat(total_sim_matrix).sort_values(by=["Target", "Correlation coefficient"], ascending=[False, False]).to_excel(os.path.join(self.result_dir, f"{when}_similarity_matrix.xlsx"))

    def Select_optimal_k(self, lower, upper, constraint_step=12):
        upper += 1
        inertias = []
        silhouettes = []
        CHS = []
        DBI = []
        print(
            f"\n------------------------------- Select optimal number of cluster -------------------------------")
        for k in tqdm(range(lower, upper)):
            tk = TimeSeriesKMeans(n_clusters=k, n_init=1, random_state=42, metric="dtw", init="k-means++",
                                  metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": constraint_step,
                                                 "n_jobs": -1},
                                  n_jobs=-1)
            tk.fit_predict(self.data)
            silhouette = silhouette_score(self.data, tk.labels_, metric="dtw", n_jobs=-1,
                                          metric_params={"global_constraint": "sakoe_chiba",
                                                         "sakoe_chiba_radius": constraint_step, "n_jobs": -1})
            chs = calinski_harabasz_score(self.data, tk.labels_)
            dbi = davies_bouldin_score(self.data, tk.labels_)
            inertias.append(tk.inertia_)
            silhouettes.append(silhouette)
            CHS.append(chs)
            DBI.append(dbi)

        plt.cla()
        plt.figure(figsize=(16, 10))
        ax1, ax2 = plt.gca(), plt.gca().twinx()
        line1 = ax1.plot(range(lower, upper), inertias, "o-", color="b", linewidth=2, label="Inertia")
        ax1.set_xlabel("K (Number of Clusters)", labelpad=10, fontsize=18)
        ax1.set_ylabel("Inertia", labelpad=10, fontsize=18)
        ax1.tick_params(axis="both", labelsize=16)
        line2 = ax2.plot(range(lower, upper), DBI, "o-", color="r", linewidth=2, label="Davies-Bouldin score")
        ax2.set_ylabel("Davies-Bouldin score", labelpad=10, fontsize=18)
        ax2.tick_params(axis="y", labelsize=16)
        ax1.legend(line1 + line2, [x.get_label() for x in (line1 + line2)], fontsize=20)
        ax1.grid(True)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "Optimal_K(inertia, DBI).png"), dpi=400)

        plt.cla()
        plt.figure(figsize=(16, 10))
        ax1, ax2 = plt.gca(), plt.gca().twinx()
        line1 = ax1.plot(range(lower, upper), CHS, "o-", color="g", linewidth=2, label="Calinski-Harabasz score")
        ax1.set_xlabel("K (Number of Clusters)", labelpad=10, fontsize=18)
        ax1.set_ylabel("Calinski-Harabasz score", labelpad=10, fontsize=18)
        ax1.tick_params(axis="both", labelsize=16)
        line2 = ax2.plot(range(lower, upper), silhouettes, "o-", color="orange", linewidth=2, label="Silhouette score")
        ax2.set_ylabel("Silhouette score", labelpad=10, fontsize=18)
        ax2.tick_params(axis="y", labelsize=16)
        ax1.legend(line1 + line2, [x.get_label() for x in (line1 + line2)], fontsize=20)
        ax1.grid(True)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "Optimal_K(CHS, silhouette).png"), dpi=400)


timeseries = []
file_names = []
# 원본 FID 이미지 디렉토리
dir = "C:/Users/user/Desktop/마약지문감정 데이터/FID_I.S 150%"

for idx, file in enumerate(os.listdir(dir)):
    if file == "Thumbs.db":
        continue
    file_names.append(file.split('.')[0])
    file_path = os.path.join(dir, file)
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image binarization
    otsu_threshold, binary_image = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Image resize
    min_y = 25
    max_y = np.where(binary_image[:, -10:] > 0)[0][-1]
    min_x = np.where(binary_image[:20, :] > 0)[1].max()
    max_x = -15
    binary_image = binary_image[min_y:max_y - 10, min_x - 10:max_x]

    # y축 gab 제거
    min_threshold = set()
    for i in range(binary_image.shape[1]):
        if len(np.where(binary_image[:, i] > 0)[0]) > 1:
            min_threshold.add(np.where(binary_image[:, i] > 0)[0][-1])
    y_threshold = np.max(list(min_threshold))
    binary_image = binary_image[:y_threshold + 5, :]
    img = cv2.resize(binary_image, (1200, 600))

    # 픽셀 강도 255로 통일
    img = np.where(img != 0, 255, img)
    # 전처리 된 FID 이미지룰 저장하고 싶을때
    # if file not in os.listdir("./processed_FID"):
    #     cv2.imwrite(f"./processed_FID/{file}", img)

    # Convert to timeseries
    timeseries.append(img.shape[0] - img.argmax(axis=0))

# 시계열 데이터로 변환된 File을 저장하고 싶을 때 (파일이름 지정 가능, 확장자는 npy로 할 것)
# file_name = "TIC_internal_scaling_150%_dataset_108_timeseries.npy"
# np.save(os.path.join(main_dir, file_name), timeseries)

obj = FID(data=timeseries, file_names=file_names)

# obj.Select_optimal_k(lower=10, upper=30, constraint_step=12)
Subsequence_matrix = obj.Sliding_window(stride=30, window_size=240)
Sim_matrix = obj.CSPA(data=Subsequence_matrix, n_clusters=21, constraint_step=12)
obj.Correlation_heat_dendrogram(data=Sim_matrix, show_figrue=False)
obj.Saving_similar_image(time_lag=12, correlation_threshold=0.9, NCC_threshold=0.95)