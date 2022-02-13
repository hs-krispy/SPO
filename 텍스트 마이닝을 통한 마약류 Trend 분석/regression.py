import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

plt.rc("font", family="Malgun gothic")
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("C:/Users/user/PycharmProjects/pythonProject/공모전/Traffic.csv", encoding="UTF-8", header=1)
label = pd.read_excel("C:/Users/user/PycharmProjects/pythonProject/공모전/연도별 마약사범 단속.xlsx", index_col=0)
label["Sum"] = label.sum(axis=1)
df.set_index("주", inplace=True)
df.columns = ["대마", "필로폰", "펜타닐"]
print(df.max(axis=0))

# 주별 Traffic을 월별로 변환 (평균치로)
df = df.iloc[:-4, :]
df = df.pivot_table(values=df.columns, index=list(map(lambda x: x[:-3], df.index)), aggfunc="mean")
df = df.apply(lambda x: x * (100 / df.max(axis=0).max()))


def line_plot(df):
    fig = px.line(df, markers=True)
    fig.update_layout(title={
        "text": "2019 ~ 2021.09 구글 검색 Traffic",
        "xanchor": "center",
        "yanchor": "top",
        "x": 0.5,
        "y": 0.95,
        "font": {"size": 40, "color": "black"}
    },
        legend={"font": {"size": 30, "color": "black"},
                "orientation": "h",
                "xanchor": "center",
                "yanchor": "bottom",
                "x": 0.5,
                "y": -0.15,
                "bgcolor": "white"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        plot_bgcolor="white")
    fig.update_xaxes(tickfont={"size": 30})
    fig.update_yaxes(tickfont={"size": 30})
    fig.update_traces(line={"width": 3})
    fig.show()

# 다중 공선성
def check_vif(x):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    vif.index = df.columns
    vif.plot.bar(rot=0, fontsize=17, width=0.2)
    plt.legend([])
    plt.show()

# shapiro wilk test (정규성 검증)
for col in df.columns:
    print(stats.shapiro(df[col]))

# 상관관계 heatmap
sns.heatmap(df.corr(method="spearman"), fmt=".2f", annot=True, square=True, cmap=plt.cm.Blues,
            annot_kws={"size": 30})


# data scaling & train test split
scaler = StandardScaler()
linear = LinearRegression()
forest = RandomForestRegressor(n_estimators=300, oob_score=True, random_state=42)

scaled_x = scaler.fit_transform(df)
train_x, test_x, train_y, test_y = train_test_split(scaled_x, label.Sum, test_size=0.3, random_state=42,
                                                    shuffle=True)
train_x = train_x.reshape(-1, 1)
test_x = test_x.reshape(-1, 1)

# 회귀분석 & 검증
def regression(model, which=None):
    model.fit(train_x, train_y)
    train_r2 = model.score(train_x, train_y)
    test_r2 = model.score(test_x, test_y)
    train_Adj_r2 = 1 - ((1 - train_r2) * ((len(train_y) - 1) / (len(train_y) - train_x.shape[1] - 1)))
    test_Adj_r2 = 1 - ((1 - test_r2) * ((len(test_y) - 1) / (len(test_y) - test_x.shape[1] - 1)))
    if which != "forest":
        df = pd.DataFrame([train_Adj_r2, test_Adj_r2, np.nan], columns=["Linear regression"],
                          index=["Train R2 (Adjusted)", "Test R2 (Adjusted)", "OOB_SCORE"])
    else:
        df = pd.DataFrame([train_Adj_r2, test_Adj_r2, model.oob_score_],
                          columns=["Random forest regression"],
                          index=["Train R2 (Adjusted)", "Test R2 (Adjusted)", "OOB_SCORE"])
    return df.T

l_res = regression(linear, "Linear")
f_res = regression(forest, "forest")
result = pd.concat([l_res, f_res], axis=0)