import os


def convert2txt(path, output_path):
    """
    文件名存储txt
    """
    files = os.listdir(path)
    with open(output_path, 'w') as f:
        for file in files:
            file_name= os.path.basename(file)
            full_name = os.path.join(path, file_name)
            f.write(full_name+'\n')


if __name__ == "__main__":
    source_file = "G:\\datasets\\Market-1501\\pytorch\\train\\0002"
    output = "F:\\ssr\\test.txt"
    convert2txt(source_file, output)
