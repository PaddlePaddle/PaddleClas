import os

def file_name(file_dir):
    temp = []
    for root, dirs, files in os.walk(file_dir):
        print(dirs) #当前路径下所有子目录
        temp = dirs #存储需要的子目录
        break
    
    # 删除不需要的子目录
    temp.remove('_templates')
    temp.remove('_build')
    temp.remove('_static')
    chinese_name = ['模型训练', '介绍', '图像识别流程', '其他', 'FAQ系列', '数据准备', '安装', '模型库', '高级教程', '算法介绍', '推理部署', '快速开始']
    # 写根目录下的rst文件
    with open('./index.rst', 'w') as f:
        f.write('欢迎使用PaddleClas图像分类库！\n')
        f.write('================================\n\n')
        f.write('.. toctree::\n')
        f.write('   :maxdepth: 1\n\n')
        for dir in temp:
            f.write('   ' + dir + '/index\n')
        f.close()

    # 写各个子目录下的rst文件
    for dir in temp:
       for root, dirs, files in os.walk(dir):
        print(root) #当前目录路径
        if 'index.rst' in files:
            files.remove('index.rst')
        print(files) #当前路径下所有非目录子文件
        curDir = os.path.join(file_dir, dir)
        filename = curDir + '/index.rst'
        idx = temp.index(dir)
        ch_name = chinese_name[idx]
        with open(filename, 'w') as f:
            f.write(ch_name+'\n')
            f.write('================================\n\n')
            f.write('.. toctree::\n')
            f.write('   :maxdepth: 2\n\n')
           
            for f1 in files:
                f.write('   ' + f1 + '\n')
            
            f.close()


def readfile(filename):
    file = open(filename)
    i = 0
    while 1:
        line = file.readline()
        print(i)
        print(line)
        i += 1
        if not line:
            break
        pass # do something
    file.close()


file_name('./')
#filename = './index.rst'
#readfile(filename)
