#coding=utf-8
from PIL import Image
import pytesseract

def main():
    # img = Image.open("/home/wxj/project/alipay.png")
    img = Image.open("/home/wxj/project/weixin.jpg")
    text = pytesseract.image_to_string(img, lang="chi_sim")
    print(text)


if __name__ == "__main__":
    main()
