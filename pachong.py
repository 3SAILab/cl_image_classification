import logging
import os
import requests
import time
 
# 设置日志配置,方便调试信息输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APPLICATION_ID = "801065"
ACCESS_KEY = "QqZbbKNDaJvXjXREy8HSp0gf7xnKRvjDyEROitapN5Q"
SECRET_KEY = "BeS3HJI9Vvy77jjBvlDEGU9WDYAwLbfSlSqDkQTM2mQ"

categories = {
    # 'daxiang': 200, 
    'eyu':200, 
    # 'gou':200, 
    # 'guihua':200, 
    # 'hehua':200, 
    # 'houzi':200,
    # 'juhua':200,
    # 'laohu': 200,
    # 'mao':200,
    # 'meigui':200,
    # 'meihua': 200,
    # 'qianniuhua': 200, 
    # 'she': 200, 
    # 'shizhuhua': 200, 
    # 'shuimu': 200, 
    # 'songshu': 200,
    # 'xiangrikui': 200, 
    # 'xiongmao': 200, 
    # 'xiuqiuhua': 200, 
    # 'yujinxiang': 200
}

category_to_english = {
    'daxiang': 'elephant',           
    'eyu': 'crocodile',            
    'gou': 'dog',                   
    'guihua': 'osmanthus',    
    'hehua': 'lotus',               
    'houzi': 'monkey',               
    'juhua': 'chrysanthemum',        
    'laohu': 'tiger',               
    'mao': 'cat',                   
    'meigui': 'rose',             
    'meihua': 'plum blossom',      
    'qianniuhua': 'morning glory',     
    'she': 'snake',            
    'shizhuhua': 'carnation',  
    'shuimu': 'jellyfish',           
    'songshu': 'squirrel',         
    'xiangrikui': 'sunflower',     
    'xiongmao': 'panda',             
    'xiuqiuhua': 'hydrangea',  
    'yujinxiang': 'tulip'         
}

data_root = os.path.abspath(os.getcwd())
data_path = os.path.join(data_root, "data", "my_data", "train")

headers = {
    "Authorization": f"Client-ID {ACCESS_KEY}",
    "Accept": "application/json"
}

def download_images():
    for category, total_images in categories.items():
        category_folder = os.path.join(data_path, category)
        os.makedirs(category_folder, exist_ok=True)

        per_page = 30  # 最大每页数量
        total_pages = (total_images + per_page - 1) // per_page  # 向上取整

        logging.info(f"开始下载 {category}（共 {total_images} 张，{total_pages} 页）")

        for page in range(1, total_pages + 1):
            # 构建搜索请求
            query = category_to_english[category]
            url = f"https://api.unsplash.com/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "page": page,
                "orientation": "landscape"  # 可选：指定图片方向
            }

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 429:
                    # 被限流，读取重置时间
                    retry_after = int(response.headers.get("Retry-After", 3600))
                    logging.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue  # 重试当前页

                elif response.status_code != 200:
                    logging.error(f"API 请求失败: {response.status_code} - {response.text}")
                    time.sleep(5)
                    continue

                data = response.json()
                results = data.get("results", [])
                logging.info(f"第 {page} 页获取到 {len(results)} 张图片")

                for photo in results:
                    img_url = photo["urls"]["regular"] 
                    img_id = photo["id"]
                    img_path = os.path.join(category_folder, f"{img_id}.jpg")

                    if os.path.exists(img_path):
                        continue  # 避免重复下载

                    try:
                        img_response = requests.get(img_url, timeout=10)
                        if img_response.status_code == 200:
                            with open(img_path, 'wb') as f:
                                f.write(img_response.content)
                        else:
                            logging.warning(f"下载失败: {img_url}")
                    except Exception as e:
                        logging.error(f"下载异常: {e}")

                    time.sleep(0.5)  

            except Exception as e:
                logging.error(f"请求异常: {e}")
                time.sleep(5)

            time.sleep(1)  

        logging.info(f"{category} 下载完成")

if __name__ == "__main__":
    download_images()

# data_path = "data/my_data/train"
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