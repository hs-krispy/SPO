import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("C:/Users/user/Desktop/약물복용 데이터 분석/practice/210906_복약데이터_통계.xlsx")

# 크레아티닌 값이 20이상 300이하에 해당하고 Dose가 0이 아닌 row 추출
df = df[(20 <= df["Concentration of creatinine (mg/dL)"]) & (df["Concentration of creatinine (mg/dL)"] <= 300) & (df["Dose (mg)"] != 0)]
# 모체농도와 대사체농도의 ND와 < C1에 대해 0 값으로 일괄 처리
df.replace(["ND", "< C1", "<C1"], 0, inplace=True)
# 따라서 보정값들도 마찬가지로 0 값으로 처리
df.loc[:, ["Normalized concentration of parent (ng/mg[Cr])", "Normalized concentration of metabolite (ng/mg[Cr])"]] = \
    df[["Normalized concentration of parent (ng/mg[Cr])", "Normalized concentration of metabolite (ng/mg[Cr])"]].fillna(0)
# 표본이 너무 적은 노년과 소년을 각각 중년과 청년 집단으로 변환
df.replace({"노년": "중년", "소년": "청년"}, inplace=True)

# 성분에 따라 데이터 분리
Quetiapine = df[df.성분명 == "Quetiapine"]
selected_columns = ["Normalized concentration of parent (ng/mg[Cr])", "Normalized concentration of metabolite (ng/mg[Cr])",
                    "Concentration of creatinine (mg/dL)", "Dose (mg)"]

order = ["M", "F"]
for sc in selected_columns:
    fig, axes = plt.subplots(figsize=(15, 7.5))
    sns.violinplot(x=sc, y="성분명", hue_order=order, hue="Sex", split=True, data=Quetiapine, inner=None, orient="h",
                       saturation=0.6)
    sns.boxplot(x=sc, y="성분명", hue="Sex", data=Quetiapine, boxprops={"zorder": 2}, width=0.3,
                hue_order=order)
    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.tick_params(axis="x", labelsize=21.5)
    handles, labels = axes.get_legend_handles_labels()
    # boxplot 범례는 제외하고 표시
    plt.legend(handles[:2], labels[:2], loc="best", title="Sex", title_fontsize=20, fontsize=20, ncol=2)
    plt.ylabel("")
    plt.xlabel(f"{sc}", fontsize=23, labelpad=18)
    plt.yticks([])
    plt.show()