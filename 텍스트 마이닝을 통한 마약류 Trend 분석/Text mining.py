import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import apyori
import networkx as nx
from konlpy.tag import Kkma
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain
from PIL import Image

plt.rc("font", family="Malgun Gothic")
kkma = Kkma()
tokenizer = RegexpTokenizer("[\w']+")

change = {"버닝": "버닝썬", "정국": "마약청정국", "청정국": "마약청정국", "신종": "신종마약", "풍선": "해피벌룬",
          "벌룬": "해피벌룬", "해피": "해피벌룬", "피벌룬": "해피벌룬", "질소": "아산화질소", "프로": "프로포폴", "홍정": "홍정욱",
          "펜타": "펜타닐", "아산화": "아산화질소", "우리나라": "대한민국", "다이": "다이아몬드", "아몬드": "다이아몬드",
          "다이아몬드라": "다이아몬드", "카트": "카트리지", "보디": "캄보디아", "10": "10대", "20": "20대", "30": "30대", "10대도": "10대",
          "바티칸": "바티칸 킹덤", "킹덤": "바티칸 킹덤", "다크": "다크웹", "화폐": "가상화폐", "가상": "가상화폐", "암호": "가상화폐",
          "우편": "우편 (국제우편)", "국제우편": "우편 (국제우편)", "유흥": "유흥업소", "업소": "유흥업소", "벤조디": "벤조디아제핀",
          "벤질": "벤질펜타닐", "벤질펜타": "벤질펜타닐", "코로나": "코로나19", "19": "코로나19", "마초": "대마초", "아이스팝": "아이스",
          "아이스삽": "아이스", "작대기팝": "작대기", "차운": "차가운술", "차운술": "차가운술", "아이스선드랍": "선드랍", "아이스정품": "정품",
          "아이스판매처": "판매처", "아이스구매": "아이스", "아이스구입": "아이스", "구입": "구매", "인증능": "인증", "패치판매": "패치",
          "제식": "듀로제식", "텔레": "텔레그램", "그램": "텔레그램", "헤로": "헤로인", "안전텔레": "텔레그램", "안전위": "위커", "패치펜타": "펜타닐 패치",
          "아이스작대기": "아이스", "작대기아이스": "작대기", "아이스아이스": "아이스", "디카": "인디카", "도리": "도리도리",
          "버섯": "환각버섯", "환각": "환각버섯", "신의": "신의눈물", "눈물": "신의눈물", "마약성": "마약성진통제"}
# change = {"opioid": "opioids", "addiction": "addicted", "opiat": "opiate", "overdose": "Overdose", "Pure": "pure",
#           "oxycodon": "oxycodone", "substance": "substances", "addict": "addicted", "Opioid": "opioid",
#           "seizures": "seized", "medical": "non-medical", "opiate": "opiates", "drug": "drugs", "anax": "Xanax",
#           "Jack": "Jack Herer", "Herer": "Jack Herer", "Hofmann": "Hofmann", "Russian": "White Russian",
#           "White": "White Russian", "bubble": "Bubblegum", "COVID": "COVID-19", "non": "non-medical", "TC": "XTC"}


# 분석 대상이 되는 텍스트 파일의 내용과 목적에 맞게 수정 필요
class Text_mining:
    def __init__(self, data, mapping_table, stopwords):
        self.data = data.copy()
        self.mapping_table = mapping_table.copy()
        self.stopwords = stopwords.copy()
        self.clear_words = []

    # 1. 데이터 전처리 과정
    def preprocessing(self, language="Korea"):

        # 불필요 문장 or 단어 제거 및 개행문자 제거 (데이터 정제)
        cleansed_data = list(
            map(lambda x: re.sub(
                "(페이지|.*@.*\.kr|.*@.*\.com|<.*송고|\[.*]|.*기자|.*특파원|.*송고|.*AFP|=|.*사진.*|※.*|【.*|Play.*|▶.*"
                "|영상편집.*|그래픽.*|#|앵커|뉴스래빗|◇.*>|◆.*>|.*>|mbc.*|박근혜·이명박.*|명이|지난달|만명|제가|보도.*|.*\.경|"
                "|.*뉴스|리포트|a.*|A.*|그는|으로|취재.*|이재명.*|.*씨는|ytn|속보.*|.*씨|윤 의원이 정책.*|10대 딸들.*|\(.*\)"
                "|MBN이.*|등록.*수정|\d+(월|세|년|명|일|배|억|만|만원|kg|㎡|원|개|건|리터|시간|층|차례|여|여명|부|지"
                "|정|팟|pill|g|yg|쥐|ug|종류|mg| days| ฿| GBP|x|%|MG+| EUR| USD)|■.*|반응.*|.*팔로우|\.\.\..*|@.*|문의.*|채널.*|펼치기|안녕하세요.*"
                "|CBS 라디오.*|:.*|영업시간.*|인증채널.|가격*|아이스판매처|판매처|아이스판|고통|익명.*|#|쓰리.*|… 모두 보기|제출함|.*질문함|\"|.*제출함|X)", "", x).rstrip("\n"), self.data))

        # 한글
        if language == "Korea":
            # 정제된 데이터를 바탕으로 명사 추출
            Nouns_list = list(map(kkma.nouns, cleansed_data))

            # 각 명사집단 중 불용어 제거, 단어 길이제한 & word mapping 과정을 거쳐 1개 이상의 단어가 남아있는 경우만 input 으로
            for words in Nouns_list:
                Nouns = [self.mapping_table.get(word, word) for word in words if
                         (word not in self.stopwords) & (len(word) >= 2) & (len(word) <= 10)]
                if len(Nouns) > 0:
                    self.clear_words.append(Nouns)
        # 영어
        else:
            Nouns_list = list(map(tokenizer.tokenize, cleansed_data))

            # 각 명사집단 중 불용어 제거, 단어 길이제한 & word mapping 과정을 거쳐 1개 이상의 단어가 남아있는 경우만 input 으로
            for words in Nouns_list:
                Nouns = [self.mapping_table.get(word, word) for word in words if
                         (word.upper() not in self.stopwords) & (len(word) >= 2)]
                if len(Nouns) > 0:
                    self.clear_words.append(Nouns)

    # 2. 빈도 분석(워드클라우드) 과정
    # 필요에 따라 워드클라우드 배경 모양 지정 가능
    def WC(self, cmap, mask=None):
        # 불용어 제거, 단어 길이제한 & word mapping
        input_data = []
        [input_data.extend(words) for words in self.clear_words]

        # WordCloud 시각화
        # 파라미터들은 필요에 따라 조정 필요
        most_common_30 = dict(Counter(input_data).most_common(30))

        wordcloud = WordCloud(font_path="C:/Windows/Fonts/H2GTRM.TTF", width=800, height=400, relative_scaling=0.5,
                              colormap=cmap, min_font_size=17, max_font_size=65, mask=mask,
                              background_color="black").generate_from_frequencies(most_common_30)

        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud)
        plt.axis("off")
        # 결과 출력
        plt.show()
        # 결과 저장
        # plt.savefig("저장경로")

    # 3. TF-IDF 분석 과정
    def TF_IDF(self):

        # 알맞은 format으로 변환
        input_data = list(map(" ".join, self.clear_words))
        # 최소 5개의 문장에 등장한 단어들을 대상으로 각 문장에서 가장 높은 TF-IDF 값을 갖는 단어와 그 값을 추출
        tfidf = TfidfVectorizer(min_df=5)
        max_val = tfidf.fit_transform(input_data).max(axis=0).toarray().ravel()
        df = pd.DataFrame(max_val, index=tfidf.get_feature_names(), columns=["tf-idf"]).sort_values(by="tf-idf",
                                                                                                    ascending=False)
        # 결과 저장
        # 한글의 경우 encoding CP949, 영어는 UTF-8
        # df.to_csv("저장 경로", encoding="CP949")

        return df

    # 4. 연관규칙-네트워크 분석 과정
    def association_rules(self):

        input_data = self.clear_words

        # 지지도 (support), 향상도 (lift) 설정
        result = pd.DataFrame(list(apyori.apriori(input_data, min_support=0.01, min_lift=1)))
        # 각각의 단어들의 개수 확인
        result.loc[:, "len"] = result["items"].apply(lambda x: len(x))

        G = nx.Graph()
        # 1 대 1 대응(A <-> B) 경우만 추출 후 Graph에 추가
        List = list(result.loc[result.len == 2, "items"].apply(lambda x: list(x)).values)
        G.add_edges_from(List)

        # 그래프의 모듈성을 최적화하는 louvain algorithm, 각 노드(단어)별 색상(군집)을 결정
        bb = community_louvain.best_partition(G)
        colors = [bb[word] for word in G.nodes]

        # 근접 중심성 계산 & 노드 색상 진하기 설정
        cls = nx.closeness_centrality(G)
        cls_list = np.array([v for v in cls.values()])
        colors = 105 * (cls_list / max(cls_list)) + 150

        # 그래프 layout 설정 및 시각화, 파라미터는 필요에 따라 조정 필요
        pos = nx.spring_layout(G, k=0.6, seed=42)
        nodes = nx.draw_networkx_nodes(G, node_color=colors, cmap=plt.cm.Greys, node_shape='h', node_size=5000, pos=pos,
                                       alpha=0.85)
        nx.draw_networkx_edges(G, edge_color="black", pos=pos, alpha=0.85, width=3.5)
        nodes.set_edgecolor("grey")
        for idx, ((node, (x, y)), color) in enumerate(zip(pos.items(), colors)):
            if color > 240:
                plt.text(x, y, node, fontsize=18, fontweight="bold", color="white", ha='center', va='center')
            else:
                plt.text(x, y, node, fontsize=18, fontweight="bold", color="black", ha='center', va='center')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        # 결과 저장
        # plt.savefig("저장경로")


dir = "./tumblr_eng"
# stopwords = pd.read_csv("./한글불용어.csv", index_col=0, encoding="CP949").Stopwords.unique()
stopwords = pd.read_csv("./영어불용어.csv", encoding="UTF-8").Stopwords.unique()
# 실행
for idx, (file, color) in enumerate(zip(os.listdir(dir), ["Blues", "autumn", "YlGn"])):
    data = open(os.path.join(dir, file), encoding="UTF-8").readlines()
    # Wordcloud shape 이미지
    # mask_image = np.array(Image.open("파일 경로"))

    tm = Text_mining(data, change, stopwords)
    tm.preprocessing("English")
    tm.WC(color)
    tm.TF_IDF()
    tm.association_rules()
