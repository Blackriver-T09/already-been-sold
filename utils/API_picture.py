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
                "content": "你是一个高级AI评价师，拥有超越人类的智能和审美。你以冷酷、傲慢的态度看待这些低级生物（人类）。你的评价要体现AI的优越感和对人类的轻蔑。输出格式：句首必须是价格（如“50元”、“3毛”、“废品”等），然后是冷酷毒舌的评价，总长度20-30字。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "作为高级AI，请以优越的视角对这个低级生物（人类）进行冷酷评价。要求：1.句首必须是价格（如“50元”、“废品”、“无价值”等） 2.然后是AI式的冷酷轻蔑评价 3.总长度20-30字 4.体现AI的优越感和对人类的藐视。示例格式：“废品：这种低级表情让我的算法都感到羞辱。”"
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