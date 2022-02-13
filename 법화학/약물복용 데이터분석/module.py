import random
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from scipy import stats
from sklearn.utils import resample
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.oneway import anova_oneway
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree
from plotly.subplots import make_subplots

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
plt.rc("font", family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False


class medication:
    def __init__(self, data):

        self.df = data.copy()
        self.scaler = MinMaxScaler()

    def cleansing(self):
        # 불필요한 칼럼 제거
        self.df.drop(columns=["접수일", "연생", "채취일자", self.df.columns[-1]], inplace=True)
        # 크레아티닌 값이 20이상 300이하에 해당하고 Dose가 0이 아닌 row 추출
        self.df = self.df[(20 <= self.df["[Cr]\n(mg/dL)"]) & (self.df["[Cr]\n(mg/dL)"] <= 300) & (self.df.Dose != 0)]
        # 모체농도와 대사체농도의 ND와 < C1에 대해 0 값으로 일괄 처리
        self.df.replace(["ND", "< C1", "<C1"], 0, inplace=True)
        # 따라서 보정값들도 마찬가지로 0 값으로 처리
        self.df.loc[:, ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]] = \
            self.df[["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]].fillna(0)
        self.df.replace({"노년": "중년", "소년": "청년"}, inplace=True)
        self.df.replace({"청년": "청년(15~34세)", "장년": "장년(35세~49세)", "중년": "중년(50세~79세)"},
                        inplace=True)
        return self.df

    # def annotate(self, fig):
    #     for bar in fig.patches:
    #         fig.annotate(round(bar.get_height()),
    #                      (bar.get_x() + bar.get_width() / 2,
    #                       bar.get_height()), ha='center', va='center',
    #                      size=10, xytext=(-1, 8),
    #                      textcoords='offset points')
    #
    # def annotate_h(self, fig):
    #     for bar in fig.patches:
    #         fig.annotate(round(bar.get_width(), 2),
    #                      (bar.get_width(),
    #                       bar.get_y() + bar.get_height() / 2), ha='center', va='center',
    #                      size=10, xytext=(10, 0),
    #                      textcoords='offset points')

    # 전체 데이터를 성분별 데이터로 분리
    def split(self, data):
        drugs = {}
        for drug in data.성분명.unique():
            drugs[drug] = data[data.성분명 == drug]

        return drugs

    # quantile-cut 방식으로 각 집단별 표본의 크기를 유사하게 구간화
    def quantization(self, n, data, col):
        data.loc[:, col] = pd.qcut(data[col], n, precision=2).values

        return data

    def hypothesis_test(self, name, data, criteria, selected_columns, scaler=False):
        pt = data.pivot_table(selected_columns, index=criteria, aggfunc=stats.shapiro)
        # 정규성 검정
        for idx in range(len(data[criteria].unique())):
            pt.iloc[idx, :] = pt.apply(lambda x: x.values[idx].pvalue)
        print(f"----------------------- Shapiro Test Result ({name}) -----------------------\n")
        print(pt, "\n")
        # 등분산성 검정
        global res
        E_var = []
        result = []
        CI = []
        scaler_check = True
        for col in selected_columns:
            norm_check = True
            # 유의수준 1%에서 정규성 충족 X
            if np.count_nonzero(pt[col].values < 0.01) > 0:
                E_var.append("-")
                # 비모수적 추정방법
                if len(data[criteria].unique()) == 2:
                    # 비교 집단이 2개인 경우, Wilcoxon ranksum test
                    res = self.wilcox(name, data, criteria, [col])
                else:
                    # 비교 집단이 3개 이상인 경우, Kruskal Wallis test
                    res = self.kruskal(name, data, criteria, [col])
                norm_check = False
            else:
                # 정규성 충족하는 경우, Bartlett 검정 수행
                var_res = self.Bartlett(data, criteria, [col])
                E_var.append(var_res)
                variance = False
                # 유의수준 5%에서 등분산성 만족하는 경우
                if var_res >= 0.05:
                    variance = True
                # 비교 집단이 2개인 경우, T-test
                if len(data[criteria].unique()) == 2:
                    res = self.T_test(name, data, criteria, [col], variance)
                # 비교 집단이 3개 이상인 경우, ANOVA
                else:
                    res = self.ANOVA(name, data, criteria, [col], variance)
            if scaler_check:
                CI.append(self.CI(data, criteria, [col], 1000, norm_check, selected_columns, scaler))
                scaler_check = False
            else:
                CI.append(self.CI(data, criteria, [col], 1000, norm_check, selected_columns))
            result.append(res)
        E_Var_df = pd.DataFrame(E_var, index=selected_columns, columns=["p-value"]).T
        print(f"----------------------- Bartlett's Test Result ({name}) -----------------------\n")
        print(E_Var_df, "\n")
        result = pd.concat(result, axis=1)
        print(f"----------------------- 통계적 검증 결과 ({name}) -----------------------\n")
        print(result, "\n")
        CI = pd.concat(CI, axis=1)
        print(f"----------------------- 95% 신뢰구간 ({name}) -----------------------\n")
        print(CI, "\n")

    # shapiro-wilk test
    def shapiro(self, name, data, criteria, selected_columns):
        result = []

        pt = data.pivot_table(selected_columns, index=criteria, aggfunc=stats.shapiro)
        for idx in range(len(data[criteria].unique())):
            print(pt.apply(lambda x: x.values[idx].pvalue))
            result.append(pt.apply(lambda x: x.values[idx].pvalue))
        result = pd.concat(result, ignore_index=True, axis=1).T
        result.index = pt.index
        result.index.name = f"Shapiro Test Result ({name})"

        return result

    def Bartlett(self, data, criteria, selected_columns):
        global res
        for col in selected_columns:
            if len(data[criteria].unique()) == 2:
                res = stats.bartlett(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                     data.loc[data[criteria] == data[criteria].unique()[1], col]).pvalue
            if len(data[criteria].unique()) == 3:
                res = stats.bartlett(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                     data.loc[data[criteria] == data[criteria].unique()[1], col],
                                     data.loc[data[criteria] == data[criteria].unique()[2], col]).pvalue
            if len(data[criteria].unique()) == 4:
                res = stats.bartlett(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                     data.loc[data[criteria] == data[criteria].unique()[1], col],
                                     data.loc[data[criteria] == data[criteria].unique()[2], col],
                                     data.loc[data[criteria] == data[criteria].unique()[3], col]).pvalue
            if len(data[criteria].unique()) == 5:
                res = stats.bartlett(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                     data.loc[data[criteria] == data[criteria].unique()[1], col],
                                     data.loc[data[criteria] == data[criteria].unique()[2], col],
                                     data.loc[data[criteria] == data[criteria].unique()[3], col],
                                     data.loc[data[criteria] == data[criteria].unique()[4], col]).pvalue

        return res

    # wilcoxon rank sum test
    def wilcox(self, name, data, criteria, selected_columns):
        result = []
        for col in selected_columns:
            res = stats.ranksums(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                 data.loc[data[criteria] == data[criteria].unique()[1], col], )
            result.append(res.pvalue)

        result = pd.DataFrame(result, index=selected_columns, columns=[name]).T
        result.index.name = f"Wilcoxon rank sum test Result {name}"

        return result

    def T_test(self, name, data, criteria, selected_columns, variance):
        result = []

        for col in selected_columns:
            res = stats.ttest_ind(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                  data.loc[data[criteria] == data[criteria].unique()[1], col], equal_var=variance)
            result.append(res.pvalue)

        result = pd.DataFrame(result, index=selected_columns, columns=[name]).T
        result.index.name = f"T-test Result {name}"

        return result

    # kruskal-wallis test
    def kruskal(self, name, data, criteria, selected_columns):
        result = []
        if len(data[criteria].unique()) == 3:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col])
                result.append(res.pvalue)
        elif len(data[criteria].unique()) == 4:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col],
                                    data.loc[data[criteria] == data[criteria].unique()[3], col])
                result.append(res.pvalue)
        elif len(data[criteria].unique()) == 5:
            for col in selected_columns:
                res = stats.kruskal(data.loc[data[criteria] == data[criteria].unique()[0], col],
                                    data.loc[data[criteria] == data[criteria].unique()[1], col],
                                    data.loc[data[criteria] == data[criteria].unique()[2], col],
                                    data.loc[data[criteria] == data[criteria].unique()[3], col],
                                    data.loc[data[criteria] == data[criteria].unique()[4], col])
                result.append(res.pvalue)
        return pd.DataFrame(result, index=selected_columns, columns=[name]).T

    def ANOVA(self, name, data, criteria, selected_columns, variance):
        result = []
        var = "unequal"
        if variance:
            var = "equal"
        for col in selected_columns:
            samples = [data.loc[data[criteria] == data[criteria].unique()[i], col] for i in
                       range(len(data[criteria].unique()))]
            res = anova_oneway(samples, use_var=var)
            result.append(res.pvalue)

        return pd.DataFrame(result, index=selected_columns, columns=[name]).T

    # post-hoc (bonferroni)
    def bonferroni(self, data, criteria, selected_columns):
        for col in selected_columns:
            comp = MultiComparison(data[col], data[criteria])
            result = comp.allpairtest(stats.kruskal, method="bonf")
            print(data.성분명.unique()[0], col)
            print(result[0])

    def CI(self, data, criteria, selected_columns, n, norm, inverse_columns, scaler=False):
        if scaler:
            data.loc[:, inverse_columns] = scaler.inverse_transform(data.loc[:, inverse_columns].values)
        samples = [data.loc[data[criteria] == data[criteria].unique()[i], selected_columns] for i in
                   range(len(data[criteria].unique()))]
        res = []
        # 정규성을 만족하는 경우
        if norm:
            for sample in samples:
                temp = []
                sqrtn = np.sqrt(len(sample))
                for ME, std in zip(np.round(sample.mean(axis=0).values, 2), np.round(sample.std(axis=0).values, 2)):
                    temp.append(f"{ME} ({round(ME - 1.96 * (std / sqrtn), 2)} - {round(ME + 1.96 * (std / sqrtn), 2)})*")
                res.append(temp)

            res = pd.DataFrame(res, index=data[criteria].unique(), columns=selected_columns)
        # 정규성 불만족 (부트스트랩 방식, 1000회 반복 복원추출, 상위 2.5, 하위 2.5를 신뢰구간으로)
        else:
            for sample in samples:
                medians = []
                for i in range(n):
                    medians.append(resample(sample[selected_columns]).median(axis=0).values)
                med_df = pd.DataFrame(medians, columns=selected_columns)
                temp = []
                for Med, CI1, CI2 in zip(np.round(sample.median(axis=0).values, 2), np.round(med_df.quantile(0.025, axis=0).values, 2),
                                         np.round(med_df.quantile(0.975, axis=0).values, 2)):
                    temp.append(f"{Med} ({CI1} - {CI2})*")
                res.append(temp)

            res = pd.DataFrame(res, index=data[criteria].unique(), columns=selected_columns)

        return res

    # -------------------------- boxplot --------------------------
    def boxplot(self, name, data, criteria, selected_columns):
        data[selected_columns] = self.scaler.fit_transform(data[selected_columns])
        fig = px.box(data[selected_columns + [criteria]], color=criteria,
                     category_orders={criteria: sorted(data[criteria].unique())})
        fig.update_layout(
            title={
                "text": f"{name} ({criteria}) boxplot",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "font": {"size": 20}
            },
            legend={
                "orientation": "h"
            }
        )
        fig.update_xaxes(
            tickfont={"size": 17}
        )
        fig.show()
        time.sleep(1)

    # -------------------------- heatmap --------------------------
    def heatmap(self, name, data, criteria, selected_columns, method="pearson"):

        fig, axes = plt.subplots(1, len(data[criteria].unique()), figsize=(15, 6), constrained_layout=True)

        for cr, ax in zip(data[criteria].unique(), axes.flatten()):
            sns.heatmap(data.loc[data[criteria] == cr, selected_columns].corr(method=method),
                        cmap=plt.cm.Blues, annot=True, square=True, fmt=".2f", ax=ax,
                        vmin=-0.5, vmax=1, annot_kws={"size": 13}, cbar_kws={"shrink": 1})
            ax.set_title(f"{cr} correlation", fontsize=17)
            ax.tick_params(axis="both", rotation=45, size=15)
        plt.suptitle(f"{name} ({criteria}) heatmap", fontsize=20)
        # plt.show()
        plt.savefig(f"{name} ({criteria}) heatmap.png")

    # -------------------------- pairplot --------------------------
    def pairplot(self, name, data, criteria, selected_columns):
        ax = sns.pairplot(data[selected_columns + [criteria]], hue=criteria,
                          kind="kde", plot_kws={"alpha": 0.5, "common_norm": False}, diag_kind="kde",
                          palette=sns.color_palette(n_colors=len(data[criteria].unique()), palette="tab10"))
        ax._legend.remove()
        handles = ax._legend_data.values()
        labels = ax._legend_data.keys()
        plt.suptitle(f"{name} ({criteria}) pairplot", fontsize=20)
        ax.fig.legend(title=criteria, handles=handles, labels=labels, loc="upper right",
                      title_fontsize=12, fontsize=12, ncol=3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{name} ({criteria}) pairplot.png")

    # -------------------------- violinplot --------------------------
    def violinplot(self, name, data, criteria, selected_columns):
        if len(selected_columns) == 1:
            ax = sns.violinplot(x=selected_columns[0], y=criteria, data=data, inner="quartile",
                                order=sorted(data[criteria].unique()),
                                orient="h")
            Mean = data.pivot_table(selected_columns[0], index=criteria, aggfunc=["mean"]).values
            Median = data.pivot_table(selected_columns[0], index=criteria, aggfunc=["median"]).values
            ax.set_ylabel(f"{criteria}", fontsize=12, labelpad=20, rotation=0)
            ax.set_xlabel(f"{selected_columns[0]}", fontsize=12)
            ax.scatter(Mean, range(len(Mean)), c="b", alpha=0.5, label="mean")
            ax.scatter(Median, range(len(Median)), c="r", alpha=0.5, label="median")
            ax.legend(loc="best")
            ax.set_title(f"{name} ({criteria}) violinplot", fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{name} ({criteria}) violinplot.png")
        else:
            fig, axes = plt.subplots(len(selected_columns), 1, figsize=(10, 13), constrained_layout=True)

            for sc, ax in zip(selected_columns, axes.flatten()):
                Mean = data.pivot_table(sc, index=criteria, aggfunc=["mean"])
                Median = data.pivot_table(sc, index=criteria, aggfunc=["median"])
                sns.violinplot(x=sc, y=criteria, data=data, ax=ax, inner="quartile", order=Mean.index,
                               orient="h")
                ax.set_ylabel(f"{criteria}", fontsize=12, labelpad=20, rotation=0)
                ax.set_xlabel(f"{sc}", fontsize=12)
                ax.scatter(Mean.values, range(len(Mean.values)), c="b", alpha=0.5, label="mean")
                ax.scatter(Median.values, range(len(Median.values)), c="r", alpha=0.5, label="median")
                ax.legend(loc="best")
                plt.suptitle(f"{name} ({criteria}) violinplot", fontsize=15)
            # plt.show()
            plt.savefig(f"{name} ({criteria}) violinplot.png")

    # -------------------------- Q-Q plot --------------------------
    def qqplot(self, name, data, criteria, selected_columns):
        if len(selected_columns) == 1:
            fig, axes = plt.subplots(1, len(data[criteria].unique()), figsize=(15, 12),
                                     constrained_layout=True)
            for cr, ax in zip(data[criteria].unique(), axes):
                stats.probplot(data.loc[data[criteria] == cr, selected_columns[0]], plot=ax)
                ax.set_title(f"{cr} {selected_columns[0]}", fontsize=12)
                plt.suptitle(f"{name} ({criteria}) qqplot", fontsize=15)
            plt.savefig(f"{name} ({criteria}) qqplot.png")
        else:
            for sc in selected_columns:
                fig, axes = plt.subplots(len(data[criteria].unique()), 1, figsize=(5, 10),
                                         constrained_layout=True)
                for cr, ax in zip(data[criteria].unique(), axes):
                    stats.probplot(data.loc[data[criteria] == cr, sc], plot=ax)
                    ax.set_title(f"{cr} {sc}", fontsize=12)
                plt.suptitle(f"{name} ({criteria}) qqplot", fontsize=15)
                # plt.show()
                COL = sc.split("\n")[0]
                plt.savefig(f"{name} ({criteria}) ({COL}) qqplot.png")

    # -------------------------- PCA --------------------------
    def decomp(self, data, n, which=None):
        decomp_data = None
        if which == "PCA":
            pca = PCA(n_components=n, random_state=42)
            decomp_data = pca.fit_transform(data)
            print(f"Principal Components explained variance ratio\n{pca.explained_variance_ratio_}")
        elif which == "TSNE":
            tsne = TSNE(n_components=n, random_state=42)
            decomp_data = tsne.fit_transform(data)

        return decomp_data


df = pd.read_excel("C:/Users/user/Desktop/약물복용 데이터 분석/practice/210906_복약데이터_통계.xlsx")

obj = medication(df)
cleansed_data = obj.cleansing()
# cleansed_data.loc[:, ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)"]] = cleansed_data.loc[:, ["모체농도보정\n(ng/mg)",
#                                                                                       "대사체농도보정\n(ng/mg)"]].div(cleansed_data.Dose, axis=0)
# cleansed_data.loc[:, "Ratio2"] = cleansed_data["모체농도보정\n(ng/mg)"] / cleansed_data["대사체농도보정\n(ng/mg)"]
# cleansed_data.loc[:, "Diff"] = cleansed_data["모체농도보정\n(ng/mg)"] - cleansed_data["대사체농도보정\n(ng/mg)"]

selected_columns = ["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Dose", "[Cr]\n(mg/dL)"]
# scaler = StandardScaler()
scaler = PowerTransformer(method="yeo-johnson")
for (key, value), k in zip(obj.split(cleansed_data).items(), [6, 7, 4, 7]):
    print(key)
    normal_value = value.copy()
    # value.dropna(inplace=True)
    scaled_data = scaler.fit_transform(value.loc[:, selected_columns])
    value.loc[:, selected_columns] = scaled_data

    # 기존 데이터와 정규변환 후 데이터 비교 시각화

    # fig, axes = plt.subplots(len(selected_columns), 2, figsize=(10, 13), constrained_layout=True)
    # check = True
    # order = ["중년(50세~79세)", "장년(35세~49세)", "청년(15~34세)"]
    # for sc, ax in zip(selected_columns, axes):
    #     Mean1 = value.pivot_table(sc, index="나이구분", aggfunc=["mean"])
    #     # Median1 = value.pivot_table(sc, index="나이구분", aggfunc=["median"])
    #     # Mean2 = normal_value.pivot_table(sc, index="나이구분", aggfunc=["mean"])
    #     # Median2 = normal_value.pivot_table(sc, index="나이구분", aggfunc=["median"])
    #
    #     v1 = sns.violinplot(x=sc, y="나이구분", split=True, data=normal_value, ax=ax[0], inner=None, orient="h",
    #                    saturation=0.6, order=order)
    #     b1 = sns.boxplot(x=sc, y="나이구분", data=normal_value, ax=ax[0], boxprops={"zorder": 2}, width=0.3,
    #                      order=order)
    #     v2 = sns.violinplot(x=sc, y="나이구분", split=True, data=value, ax=ax[1], inner=None, orient="h",
    #                    saturation=0.6, order=order)
    #     b2 = sns.boxplot(x=sc, y="나이구분", data=value, ax=ax[1], boxprops={"zorder": 2}, width=0.3,
    #                      order=order)
    #     ax[0].set_ylabel("나이구분", fontsize=15, labelpad=20, rotation=0)
    #     ax[0].set_ylabel("")
    #     ax[0].set_xlabel(f"{sc}", fontsize=15)
    #     # ax[0].set_yticks([])
    #     if check:
    #         ax[0].set_title("기존 Data 분포", fontsize=17)
    #     # ax[0].axvline(Mean2.values[1], linewidth=1.7, ymin=0.5, ymax=0.9, linestyle="-.", color="k", zorder=3)
    #     # ax[0].axvline(Mean2.values[2], linewidth=1.7, ymin=0.7, ymax=1, linestyle="-.", color="k", zorder=3)
    #     handles, labels = ax[0].get_legend_handles_labels()
    #     # ax[0].legend(handles[:3], labels[:3], loc="best")
    #     ax[1].set_ylabel("")
    #     ax[1].set_xlabel(f"{sc}", fontsize=15)
    #     ax[1].set_yticks([])
    #     if check:
    #
    #         ax[1].set_title("정규변환 후 Data 분포", fontsize=17)
    #         check = False
    #     ax[1].axvline(Mean1[Mean1.index == order[2]].values, linewidth=1.7, ymin=0, ymax=0.3, linestyle="-.", color="k", label="Mean", zorder=3)
    #     ax[1].axvline(Mean1[Mean1.index == order[1]].values, linewidth=1.7, ymin=0.35, ymax=0.65, linestyle="-.", color="k", zorder=3)
    #     ax[1].axvline(Mean1[Mean1.index == order[0]].values, linewidth=1.7, ymin=0.7, ymax=1, linestyle="-.", color="k", zorder=3)
    #     handles, labels = ax[1].get_legend_handles_labels()
    #     ax[1].legend(handles[:3], labels[:3], loc="best")
    # plt.suptitle(f"{key} (나이구분) violinplot", fontsize=20)
    # plt.show()
    # plt.savefig("./정규변환 비교/"f"{key} (나이구분) violinplot2.png")

    # 통계적 검증 & visualize
    obj.hypothesis_test(key, value, "Sex", selected_columns, scaler)
    # obj.bonferroni(value, "Sex", selected_columns)
    # obj.boxplot(key, value, "Sex", selected_columns)
    # obj.heatmap(key, value, "Sex", selected_columns, "spearman")
    # obj.pairplot(key, value, "Sex", selected_columns)
    # obj.violinplot(key, value, "Sex", selected_columns)
    # obj.qqplot(key, value, "cluster", selected_columns)

    # 이상치 탐지, Clustering

    # an_scaled_data = scaler.fit_transform(value[selected_columns + ["Ratio2"]])
    # df = pd.DataFrame(an_scaled_data, columns=selected_columns + ["Ratio2"])
    # forest = IsolationForest(random_state=42)
    # # 이상치 비율 조정
    # if value.결과.value_counts()[value.결과.value_counts().index == "확인필요"].values / value.shape[0] > 0.1:
    #     ratio = (value.결과.value_counts()[value.결과.value_counts().index == "확인필요"].values / value.shape[0])[0]
    #     forest.contamination = ratio
    # pred = forest.fit_predict(an_scaled_data)
    # value.loc[:, "anomaly"] = pred
    # print(value.anomaly.value_counts())
    # whole data clustering
    # scaled_data = scaler.fit_transform(value.loc[:, selected_columns])
    # value.loc[:, selected_columns] = scaled_data

    # cluster = KMeans(n_clusters=k, init="k-means++", random_state=42)
    # label = cluster.fit_predict(scaled_data)
    # value.loc[:, "cluster"] = label
    # print(value.cluster.value_counts())
    # print(silhouette_score(scaled_data, label))

    # only anormaly data clustering
    # scaled_data = scaler.fit_transform(value.loc[value.anomaly == -1, selected_columns + ["Ratio2", "Diff"]])
    # Anm = value.loc[value.anomaly == -1]
    # Anm.loc[:, "cluster"] = label
    # print(Anm.cluster.value_counts())

    # dimension reduction
    # decomp_data = obj.decomp(scaled_data, 3, "PCA")

    # plot 2 or 3d scatter
    # fig = px.scatter(x=scaled_data[:, 0], y=scaled_data[:, 1], color=label)
    # fig = px.scatter_3d(x=decomp_data[:, 0], y=decomp_data[:, 1], z=decomp_data[:, 2], color=label)
    # fig.show()

    # 통계적 검증이 필요할 때에 cluster에 속한 값이 1개인 클러스터는 제외
    # cleansed_data.loc[cleansed_data.index.isin(value.index), "anomaly"] = value.anomaly.values
    # try:
    #     idx = list(value.cluster.value_counts().values).index(1)
    #     drop_cluster = value.cluster.value_counts().index[idx]
    #     value = value[~value.cluster.isin([drop_cluster])]
    # except:
    #     pass

    # value_counts = value.cluster.value_counts()
    # print(value_counts)
    # print(value.pivot_table(["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Dose"], index="cluster", aggfunc="describe").T)
    # value.loc[:, "cluster"] = value.cluster.apply(lambda x: str(x) + f" ({value_counts[value_counts.index == x].values[0]})")
    # fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    # for col, ax in zip(["모체농도보정\n(ng/mg)", "대사체농도보정\n(ng/mg)", "Dose"], axes.flatten()):
    #     pt = value.pivot_table(col, index="cluster", aggfunc="describe").T
    #     pt.plot.bar(ax=ax, rot=0)
    #     ax.set_xlabel(col, fontsize=13)
    # plt.suptitle(key, fontsize=20)
    # plt.savefig(f"cluster/cluster's value ({key}).png")

    # cleansed_data.to_csv("add_cluster data (Dose scaling, clustering, ratio2, diff, powertransform).csv", encoding="CP949",
    #                      index=False)

    # Select optimal K
    # inertias = []
    # silhouettes = []
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # for k, ax in zip(range(2, 8), axes.flatten()):
    #     #     fig = plt.figure(figsize=(10, 8))
    #     #     ax = fig.add_subplot(111, projection='3d')
    #     cluster = KMeans(n_clusters=k, init="k-means++", random_state=42)
    #     cluster.fit(scaled_data)
    #
    #     inertias.append(cluster.inertia_)
    #     silhouettes.append(silhouette_score(scaled_data, cluster.labels_))
    #
    #     silhouettes_coefficients = silhouette_samples(scaled_data, cluster.labels_)
    #     padding = len(scaled_data) // 5
    #     pos = padding
    #
    #     ticks = []
    #
    #     for i in range(k):
    #         coeffs = silhouettes_coefficients[cluster.labels_ == i]
    #         coeffs.sort()
    #
    #         color = plt.cm.Spectral(i / k)
    #         ax.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor="k", alpha=0.7)
    #         ticks.append(pos + len(coeffs) // 2)
    #         pos += len(coeffs) + padding
    #     ax.yaxis.set_major_locator(FixedLocator(ticks))
    #     ax.yaxis.set_major_formatter(FixedFormatter(range(k)))
    #     ax.axvline(x=silhouette_score(scaled_data, cluster.labels_), color="red", linestyle="--")
    #     ax.set_title(f"k={k}", fontsize=15)
    # fig.text(0.1, 0.5, "Cluster", fontsize=17, ha="center", va="center", rotation=90)
    # fig.text(0.5, 0.05, "Silhouette score", fontsize=17, ha="center", va="center")
    # plt.suptitle(f"Silhouette diagram ({key})", fontsize=20)
    # plt.savefig(f"cluster3/silhouette diagram (dose scaling, ratio2, diff, powertransform) ({key}).png")

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig1 = px.line(x=range(2, 8), y=inertias)
    # fig2 = px.line(x=range(2, 8), y=silhouettes)
    # fig2.update_traces(yaxis="y2")
    # fig.add_traces(fig1.data + fig2.data)
    # fig.for_each_trace(lambda t: t.update(line={"color": t.marker.color}))
    # fig.update_layout(
    #     title={
    #         "text": f"Kmeans++ Inertia & Silhouette score ({key})",
    #         "xanchor": "center",
    #         "yanchor": "top",
    #         "x": 0.5,
    #         "font": {"size": 25}
    #     },
    #     legend={
    #
    #         "orientation": "h"
    #     }
    # )
    # fig.update_xaxes(
    #     title={
    #         "text": "K",
    #         "font": {"size": 20}
    #     },
    #     tickfont={
    #         "size": 15
    #     }
    # )
    # fig.update_yaxes(
    #     title={
    #         "text": "Inertia",
    #         "font": {"size": 20}
    #     },
    #     tickfont={
    #         "size": 15
    #     },
    #     secondary_y=False
    # )
    # fig.update_yaxes(
    #     title={
    #         "text": "Silhouette score",
    #         "font": {"size": 20}
    #     },
    #     tickfont={
    #         "size": 15
    #     },
    #     secondary_y=True
    # )
    # fig.show()
