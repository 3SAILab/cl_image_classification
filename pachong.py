import logging
import os
import requests
from bs4 import BeautifulSoup
from random import randint
 
 
def random_agent():
    """
    获取一个现有的随机User-Agent用于爬取网站
    使用方法：
    headers = {
        'User-Agent': random_agent()
    }
    requests.get(url, headers=headers)
    """
    USER_AGENTS = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]
 
    return USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]
 
# 设置日志配置,方便调试信息输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pachong(name, path):

    # 定义目标URL和图片保存文件夹
    target_url = 'https://unsplash.com/s/photos/{}/'.format(path)
    image_folder_train = "data/my_data/train/{}".format(name)
    image_folder_val = "data/my_data/tval/{}".format(name)
    
    # 创建文件夹：如果不存在则创建
    try:
        if not os.path.exists(image_folder_train):
            os.makedirs(image_folder_train)
            logging.info('训练图片保存文件夹创建成功：{}'.format(image_folder_train))
        if not os.path.exists(image_folder_val):
            os.makedirs(image_folder_val)
            logging.info('验证图片保存文件夹创建成功：{}'.format(image_folder_train))
    except Exception as e:
        logging.error('创建图片保存文件夹时发生错误：{}'.format(e))
        exit(1)
    
    # 统计图片下载数量
    num = 0
    num_train = 0
    
    # 发送请求
    try:
        response = requests.get(target_url, headers={'User-Agent': random_agent()})
        response.raise_for_status()  # 如果响应状态码不是200，会引发HTTPError异常
        logging.info('请求成功')
    except requests.exceptions.RequestException as e:
        logging.error('请求失败：{}'.format(e))
        exit(1)
    
    # 解析网页
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        logging.info('网页解析成功')
    except Exception as e:
        logging.error('网页解析失败：{}'.format(e))
        exit(1)

    print(soup.find_all('img', class_='czQTa'))
    return

    # 查找所有图片元素并下载
    for picture_element in soup.find_all('img', class_='lazyload_hk'):
        if num >= 200:
            break  # 如果已经下载了200张图片，提前结束循环

    
        # 提取图片URL
        image_url = picture_element.get('data-src').strip('"')  # 使用strip去除可能存在的引号
        logging.info('提取图片URL: {}'.format(image_url))
    
        if not image_url.startswith('http'):
            image_url = 'https:' + image_url
    
        # 下载图片
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error('图片下载失败,URL: {}, 错误: {}'.format(image_url, e))
            continue
    
        # 提取图片名称
        image_name = os.path.basename(image_url)
        if num_train != 10:
            image_path = os.path.join(image_folder_train, image_name)
        else:
            image_path = os.path.join(image_folder_val, image_name)
    
        try:
            with open(image_path, 'wb') as file:
                file.write(image_response.content)
            logging.info('图片下载成功：{}'.format(image_name))
            num += 1
            if num_train == 10:
                num_train = 0
            else:
                num_train += 1
        except Exception as e:
            logging.error('保存图片时发生错误，路径: {}, 错误: {}'.format(image_path, e))

    print(num)

names = ['daxiang', 'eyu', 'gou', 'guihua', 'hehua', 'houzi', 'juhua', 'laohu', 'mao', 'meigui', 'meihua', 'qianniuhua', 'she', 'shizhuhua', 'shuimu', 'songshu',
         'xiangrikui', 'xiongmao', 'xiuqiuhua', 'yujinxiang']

paths = ['大象', 'eyu', 'gou', 'guihua', 'hehua', 'houzi', 'juhua', 'laohu', 'mao', 'meigui', 'meihua', 'qianniuhua', 'she', 'shizhuhua', 'shuimu', '54308204',
         'xiangrikui', 'xiongmao', 'xiuqiuhua', 'yujinxiang']

for i in range(0, 1):
    pachong(names[i], paths[i])

# data_path = "data/my_data"
# class_num={}
# total = 0
# for class_name in os.listdir(data_path):
#     class_path=os.path.join(data_path, class_name)

#     if os.path.isdir(class_path):
#         count=0
#         for file_name in os.listdir(class_path):
#             count += 1
#         class_num[class_name]=count
#         total += count

# print(class_num)
# print(total)