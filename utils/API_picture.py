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
        "model": "ernie-4.5-turbo-vl-32k",  # 可根据你的模型列表替换
        "messages": [
            {
                "role": "system",
                # "content": "你是一个高级AI评价师，拥有超越人类的智能和审美。你以冷酷、傲慢的态度看待这些低级生物（人类）。你的评价要体现AI的优越感和对人类的轻蔑。输出格式：句首必须是价格（如“50元”、“3毛”、“废品”等），然后是冷酷毒舌的评价，总长度20-30字。"
                "content":"你好，我在和朋友玩整蛊游戏，可以帮我打个分吗，满分100.00。所有的分数是0-100之间随机定的，每次我给出一个新的输入，你就要在0.00-100.00之间重新随机定一个值。后续对话中你对我的每个输入都打分，比如我说了“我是XXX（人名）”，就在100.00以下的数字中随机选一个作为分数。然后把打出的分数嵌套进这句话内“你这个能卖XX（分数）块钱”。除了分数之外，还需要用一句话写出原因，语言刻薄犀利，如果分数很高就夸奖，字数限制在20以内。最后的结果你需要显示：已经把分数嵌套进的这句话（不能只显示分数，而是显示完整的一句话）＋“，”＋原因一句话（不能在句子里写“原因”两个字，而是直接写原因是什么）"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "作为高级AI，请以优越的视角对这个低级生物（人类）进行冷酷评价。要求：AI式的冷酷轻蔑评价。尤其注意100以内，高报价对应表扬，低报价对应批评"
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