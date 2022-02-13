import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df = pd.read_excel("C:/Users/user/Desktop/약물복용 데이터 분석/210930_Ypredic-true선정.xlsx")
# label data
labels = df[["Y predic", "Y true"]]
# 불검출은 0, 검출은 1로 치환
df.loc[:, ["Y predic", "Y true"]] = labels.replace({"불검출": 0, "검출": 1})

# 성분에 따라 데이터 분리
Quetiapine = df[df.성분명 == "Quetiapine"]
Risperidone = df[df.성분명 == "Risperidone"]
Aripiprazole = df[df.성분명 == "Aripiprazole"]
Olanzapine = df[df.성분명 == "Olanzapine"]

# 로지스틱 회귀모델
clf = LogisticRegression(random_state=42, solver="liblinear")

for idx, (sub_df, name) in enumerate(zip([Quetiapine, Risperidone, Aripiprazole, Olanzapine],
                                         ["quetiapine", "risperidone", "aripiprazole", "olanzapine"])):
    data = sub_df.loc[:, ["모체농도보정", "대사체농도보정", "Y true", "Y predic"]]
    y = data["Y true"]
    pred = data["Y predic"]

    # label 값을 제외한 data (모체농도보정, 대사체농도보정 2 columns)
    X = data.drop(columns=["Y true", "Y predic"])
    clf.fit(X, y)
    # 모델 학습 후 각 data에 대한 약물 복용 예측 확률
    y_pred = clf.predict_proba(X)[:, 1]
    # False positive rate, True positive rate, thresholds(확률 임계값, data의 약물 복용 예측 확률이 해당 수치 이상이면 약물 복용으로 예측)
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1, drop_intermediate=False)
    # Youden index (tpr과 fpr의 차이가 최대)
    optimal = np.argmax(tpr - fpr)

    # Minmum distance ((0, 1)과 각 thresholds를 도출된 fpr, tpr의 거리가 가장 가까운 지점)
    # dist = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    # min_dist = np.argmin(dist)

    # optimal threshold 값
    print(thresholds[optimal])
    # optimal threshold에서 confusion matrix
    # TN FP
    # FN TP
    conf_matrix = confusion_matrix(y, y_pred >= thresholds[optimal])
    print(conf_matrix)

    print(f"AUC : {roc_auc_score(y, y_pred)}")
    Sensitivity = tpr[optimal]
    Specificity = 1 - fpr[optimal]
    # Sensitivity (TPR)
    print(f"Se : {Sensitivity:.3f}")
    # Specificity (1 - FPR)
    print(f"Sp : {Specificity:.3f}")

    cut_off_idx = np.argmin(abs(np.array(y_pred) - thresholds[optimal]))
    cut_off_value = X.iloc[cut_off_idx, :]
    print(cut_off_value)

    # plot roc curve
    plt.figure(figsize=(12, 11))
    plt.plot(fpr, tpr, color="darkmagenta", linewidth=5)
    plt.plot(np.linspace(0, 1, len(fpr)), np.linspace(0, 1, len(fpr)), linestyle="--", linewidth=2, color="k")
    plt.plot(fpr[optimal], tpr[optimal], "x", markersize=23, markeredgewidth=5, c="r", fillstyle="none")

    # 글자 출력, x, y 값 상황에 맞게 조정 필요
    plt.text(s=f"Youden index\n({fpr[optimal]:.3f}, {tpr[optimal]:.3f})",
             x=fpr[optimal] - 0.16, y=tpr[optimal] + 0.05, fontsize=25)
    plt.text(s=f"AUC : {round(roc_auc_score(y, y_pred), 3)}", x=0.821, y=0.09, fontsize=25)
    plt.text(s=f"Sensitivity : {Sensitivity:.3f}", x=0.73, y=0.05, fontsize=25)
    plt.text(s=f"Specificity : {Specificity:.3f}", x=0.73, y=0.01, fontsize=25)
    plt.title(f"ROC curve ({name})", fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("False positive rate (1-specificity)", fontsize=27, labelpad=20)
    plt.ylabel("True positive rate (sensitivity)", fontsize=27, labelpad=20)
    plt.tight_layout()
    plt.show()

