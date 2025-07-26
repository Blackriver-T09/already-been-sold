import requests
import os



def generate_voice(text, filename):
    API_URL = "http://test.heihet09.com/synthesize"

    filename = os.path.join("output_voice", "voice_" + filename)
    

    response = requests.post(
        API_URL,
        json={
            "text": text,
            "filename": filename  # 可选，不写默认是 output.wav
        },
        stream=True  # 开启流式传输
    )

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 音频保存成功为 {filename}")
    else:
        print(f"❌ 请求失败，状态码：{response.status_code}")
        print(response.text)


if __name__ == "__main__":
    TEXT = "“眼镜男：雨防尘，但看起来更搞笑。”估价：1/10。"
    FILENAME = "dingzhen_test.wav"
    generate_voice(TEXT, FILENAME)
