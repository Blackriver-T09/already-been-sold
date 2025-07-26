import requests
import json
import base64
import os

API_KEY = "bce-v3/ALTAK-dETzYYwOvjIMAiLKwaKDi/ca05415926afa07923dfe96488925ebd11b515e4"


# 上传本地图片并转成 base64 图片 URL 格式
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_data}"

# 调用千帆视觉 API
def generate_poisonous_comment(image_path):

    url = "https://qianfan.baidubce.com/v2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    image_url = encode_image_to_base64(image_path)

    payload = {
        "model": "deepseek-vl2",  # 可根据你的模型列表替换
        "messages": [
            {
                "role": "system",
                "content": "你是一个毒舌幽默的AI记者，专门根据图片对人物进行荒谬搞笑的短评与估价，风格讽刺、无厘头，每句话不超过30字。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请根据这张图像中的人物，生成一句毒舌短评，必须搞笑并估价，总长度限制在30字以内,一定要包含估价！"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0.9,
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()

    try:
        content = result["choices"][0]["message"]["content"]
        print( content)
        return content
    except Exception as e:
        print("❌ 出错：", result)
        return None

# 示例调用
if __name__ == "__main__":
    image_path = "./2.png"  # 替换为你的图片路径
    generate_poisonous_comment(image_path)