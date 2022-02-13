from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# 09.16일 기준 2021년 빈도수 상위 15개 뉴스사
select_news = {"뉴스1": "//*[@id='content']/div[1]/div[1]/div[3]/article",
               "KBS": "//*[@id='cont_newstext']",
               "머니투데이": "//*[@id='textBody']",
               "내일신문": "//*[@id='contents']/p",
               "세계일보": "//*[@id='article_txt']/article",
               # tag_name
               "아시아경제": "p",
               "한국경제": "//*[@id='articletxt']",
               "뉴시스": "//*[@id='content']/div[1]/div[1]/div[3]/article",
               "서울신문": "//*[@id='articleContent']",
               "YTN": "//*[@id='CmAdContent']/span",
               "MBC": "//*[@id='content']/div/section[1]/article/div[2]/div[5]",
               "동아일보": "//*[@id='content']/div/div[1]",
               "MBN": "//*[@id='newsViewArea']",
               "연합뉴스": "//*[@id='articleWrap']/div[2]/div/div/article",
               "노컷뉴스": "//*[@id='pnlContent']"}

# 가상 browser option 설정
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# Tor browser (Dark Web) crawling 시 활성화
# options.add_argument("--proxy-server=socks5://127.0.0.1:9150")

browser = webdriver.Chrome("./chromedriver_win32 (1)/chromedriver.exe", options=options)
url = "http://antidrug.drugfree.or.kr/page/?mIdx=194&page=106"
# url open
browser.get(url)

# crawling한 data를 write할 file open
with open("article.txt", "a", encoding="UTF-8") as file:
    check = True
    while check:
        article_list = browser.find_elements_by_xpath(
            '//*[@id="section"]/section/div/article/div[3]/div[2]/table/tbody/tr')
        time.sleep(3)
        for article in article_list:
            info = article.find_element_by_xpath("./td[2]/a")
            time.sleep(2)
            title = info.text
            news = article.find_element_by_xpath("./td[3]").text
            day = article.find_element_by_class_name('day').text
            print(f"Title : {title}")
            print(f"뉴스사 : {news}")
            print(f"작성일자 : {day}")
            # 사전에 선택한 15개의 뉴스사가 아니면 스킵
            if news not in list(select_news.keys()):
                continue
            # 지정된 링크로 이동
            info.click()
            time.sleep(3)
            # 활성탭 변경
            browser.switch_to.window(browser.window_handles[-1])
            time.sleep(3)

            # 기사 내용 추출
            if news == "아시아경제":
                contents = browser.find_elements_by_tag_name("p")
                file.write(title + "\n")
                for content in contents:
                    print(content.text)
                    file.write(content.text)
            else:
                try:
                    file.write(title + "\n")
                    content = browser.find_element_by_xpath(select_news[news])
                    file.write(content.text)
                except:
                    browser.close()
                    time.sleep(3)
                    browser.switch_to.window(browser.window_handles[0])
                    continue
            time.sleep(3)
            # 탭 종료 후 기존 탭으로 활성탭 변경
            browser.close()
            time.sleep(3)
            browser.switch_to.window(browser.window_handles[0])
            time.sleep(3)
        # page 넘김
        browser.find_element_by_xpath('// *[ @ id = "section"] / section / div / article / div[3] / div[3] / a[12]').click()
        time.sleep(3)

