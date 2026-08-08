import openai
import base64
import os
from dotenv import load_dotenv
import glob

# ===================== 基础配置 =====================
# 1. 加载通义千问API Key（建议在.env文件中配置，格式：DASHSCOPE_API_KEY=你的密钥）
load_dotenv()
client = openai.OpenAI(
    api_key="your api key",  # 替换为你的通义千问API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问兼容接口
)

# 2. 配置文件夹路径（替换为你的图片文件夹路径，不要有中文/空格）
IMAGE_FOLDER = r"your path"  # 存放待识别图片的文件夹
OUTPUT_FILE = r"your path\ocr_result.txt"  # OCR结果保存的文本文件

# 3. 支持的图片格式（可根据需要添加，如.bmp/.tiff）
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".gif")


# ===================== 核心函数 =====================
def image_to_base64(image_path):
    """将本地图片转换为Base64编码"""
    try:
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        return base64_data
    except Exception as e:
        print(f"【错误】图片转Base64失败 → {image_path}：{str(e)}")
        return None


def single_image_ocr(image_path):
    """识别单张图片的文字"""
    # 转换图片为Base64
    img_base64 = image_to_base64(image_path)
    if not img_base64:
        return None

    try:
        # 调用通义千问多模态模型
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 通义千问多模态模型（支持图片识别）
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "请识别图片中的全部文字。要求：1. 保留标题层级；2. 保留列表结构；3. 保留表格行列关系；4. 不要添加图片中不存在的信息；5. 只输出OCR结果。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            temperature=0.0,  # 固定输出，保证OCR结果稳定
            max_tokens=4000  # 增大token上限，适配长文本识别
        )
        # 提取识别结果
        ocr_text = completion.choices[0].message.content.strip()
        return ocr_text
    except Exception as e:
        print(f"【错误】识别图片失败 → {image_path}：{str(e)}")
        return None


def batch_ocr_images():
    """批量处理文件夹中的所有图片"""
    # 1. 检查图片文件夹是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"【错误】图片文件夹不存在 → {IMAGE_FOLDER}")
        return

    # 2. 获取文件夹中所有支持的图片文件
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, f"*{ext}")))
        #image_files.extend(glob.glob(os.path.join(IMAGE_FOLDER, f"*{ext.upper()}")))  # 匹配大写后缀（如.JPG）

    if not image_files:
        print(f"【提示】文件夹中未找到支持的图片格式 → {SUPPORTED_FORMATS}")
        return
    image_files.sort()
    # 3. 批量识别并保存结果
    total = len(image_files)
    success_count = 0
    fail_count = 0

    print(f"【开始】共发现 {total} 张图片，开始批量识别...")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for idx, img_path in enumerate(image_files, 1):
            print(f"\n【进度】{idx}/{total} → 正在识别：{img_path}")

            # 识别单张图片
            ocr_result = single_image_ocr(img_path)
            if ocr_result:
                success_count += 1
                # 写入结果识别文字
                f.write(ocr_result + "\n\n")
                print(f"【成功】识别完成 → 已保存结果")
            else:
                fail_count += 1
                print(f"【失败】识别失败 → 跳过该图片")

    # 4. 输出统计结果
    print(f"\n【完成】批量识别结束！")
    print(f"总计：{total} 张 | 成功：{success_count} 张 | 失败：{fail_count} 张")
    print(f"结果已保存至：{OUTPUT_FILE}")


# ===================== 执行批量识别 =====================
if __name__ == "__main__":
    batch_ocr_images()