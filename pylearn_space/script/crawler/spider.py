# 爬取漫画
import re
import os
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from contextlib import closing  # 设置头


# 创建保存目录
save_dir = r'D:/个人妙妙屋/资源'
save_title = '妖神记'
save_path = os.path.join(save_dir, save_title)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

target_url = "https://www.dmzj.com/info/yaoshenji.html"  # html(info页面)
html = requests.get(target_url).content.decode()  # 解码, 默认utf-8
soup = BeautifulSoup(html, "lxml")  # 解析页面

# 获取名称列表
list_con_li = soup.find("ul", attrs={"class": "list_con_li"})  # 外部tag: ul
comic_list = list_con_li.find_all("a")  # 内部tag: li -> a
chapter_name = []
chapter_urls = []
# 遍历
for comic in comic_list:
    href = comic.get('href')  # 链接
    name = comic.text  # 章节标题
    chapter_name.insert(0, name)
    chapter_urls.insert(0, href)


# 下载漫画
for i, url in enumerate(tqdm(chapter_urls)):
    download_header = {'Referer': url}  # 设置访问头
    name: str = chapter_name[i]
    while '.' in name:
        name = name.replace('.', ' ')
    chapter_save_dir = os.path.join(save_path, name)
    if name not in os.listdir(save_path):
        os.mkdir(chapter_save_dir)
        r = requests.get(url)
        solo_html = BeautifulSoup(r.text, "lxml")
        # 解析内部动态加载
        script_info = solo_html.script
        pics = re.findall(r"\d{13,14}", str(script_info))  # 找出所有长度13-14的串符
        for idx, pic in enumerate(pics):
            if len(pic) == 13:
                pics[idx] = pic + "0"  # 13位补零
        pics = sorted(pics, key=lambda x: int(x))  # 排序同时[str->int]
        chapterpic_hou = re.findall('\|(\d{5})\|', str(script_info))[0]  # 后序号
        chapterpic_qian = re.findall('\|(\d{4})\|', str(script_info))[0]  # 前序号
        for idx, pic in enumerate(pics):
            # 补零的
            if pic[-1] == "0":
                url = (
                    "https://images.dmzj.com/img/chapterpic/"
                    + chapterpic_qian
                    + "/"
                    + chapterpic_hou
                    + "/"
                    + pic[:-1]
                    + ".jpg"
                )

            else:
                url = (
                    "https://images.dmzj.com/img/chapterpic/"
                    + chapterpic_qian
                    + "/"
                    + chapterpic_hou
                    + "/"
                    + pic
                    + ".jpg"
                )
            pic_name = '%03d.jpg' % (idx + 1)  # 名称
            pic_save_path = os.path.join(chapter_save_dir, pic_name)  # 文件路径
            with closing(
                requests.get(url, headers=download_header, stream=True)
            ) as response:
                chunk_size = 1024
                content_size = int(response.headers['content-length'])
                if response.status_code == 200:
                    with open(pic_save_path, 'wb') as file:
                        for data in response.iter_content(chunk_size=chunk_size):
                            file.write(data)
                else:
                    print('链接异常')
        time.sleep(10)
