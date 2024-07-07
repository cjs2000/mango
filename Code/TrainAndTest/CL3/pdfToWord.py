from pdf2docx import Converter


def pdf_to_word(pdf_file, word_file):
    # 创建一个转换器对象
    cv = Converter(pdf_file)

    # 执行转换
    cv.convert(word_file, start=0, end=None)

    # 关闭转换器
    cv.close()

    print(f"{pdf_file} has been successfully converted to {word_file}")


# 示例使用
pdf_file = r'C:\Users\19372\Desktop\Mango\[23]-EMA-2305.13563v2.pdf'
word_file = 'example.docx'

pdf_to_word(pdf_file, word_file)
