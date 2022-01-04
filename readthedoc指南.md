1. 注册ReadtheDocs并连接到github

2. 在github上将项目克隆到本地

3. 在本地仓库中安装Sphinx

   ```shell
   pip install sphinx
   ```

4. 创建工程

   ```shell
   sphinx-quickstart
   ```

5. 对工程进行配置

   5.1 更改主题

   在source/conf.py中更改或添加如下代码

   ```python
   import sphinx_rtd_theme
   html_theme = "sphinx_rtd_theme"
   html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
   ```

   5.2 添加markdown支持和markdown表格支持

   首先需要安装recommonmark和sphinx_markdown_tables

   ```shell
   pip install recommonmark
   pip install sphinx_markdown_tables
   ```

   在source/conf.py中更改或添加如下代码

   ```python
   from recommonmark.parser import CommonMarkParser
   source_parsers = {
       '.md': CommonMarkParser,
   }
   source_suffix = ['.rst', '.md']
   extensions = [
        'recommonmark',
        'sphinx_markdown_tables'
    ]
   ```

   以上五步具体效果可以参考https://www.jianshu.com/p/d1d59d0cd58c

6. 在创建好项目以后，根目录下应该有如下几个文件：

   - **Makefile**：在使用 `make` 命令时，可以使用这些指令（e.g. `sphinx-build`）来构建文档输出。
   - **_build**：这是触发特定输出后用来存放所生成的文件的目录。
   - **_static**：所有不属于源代码（e.g. 图片）一部分的文件均存放于此处，稍后会在构建目录中将它们链接在一起。
   - **conf.py**：用于存放 Sphinx 的配置值，包括在终端执行 `sphinx-quickstart`时选中的那些值。
   - **index.rst**：文档项目的 root 目录。如果将文档划分为其他文件，该目录会连接这些文件

7. **编写文档**：在 index.rst 文件中的主标题之后，有一个内容清单，其中包括 `toctree` 声明，它将所有文档链接都汇集到 Index。

   以根目录下的index.rst为例：

   ```rst
   欢迎使用PaddleClas图像分类库！
   ================================
   
   .. toctree::
      :maxdepth: 1
   
      models_training/index
      introduction/index
      image_recognition_pipeline/index
      others/index
      faq_series/index
      data_preparation/index
      installation/index
      models/index
      advanced_tutorials/index
      algorithm_introduction/index
      inference_deployment/index
      quick_start/index
   ```

   可以用下面的python代码实现根目录和各个子目录下的`index.rst`文件的编写

   注意：此代码应该在需要生成文档书的文件夹根目录上运行

   ```python
   import os
   
   def file_name(file_dir):
       temp = []
       for root, dirs, files in os.walk(file_dir):
           print(dirs) #当前路径下所有子目录
           temp = dirs #存储需要的子目录
           break
       
       # 删除不需要的子目录
       temp.remove('images')
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
   # filename = './index.rst'
   # readfile(filename)
   ```

8. 生成文档

   运行 `make html` 命令

9. 使用浏览器查看在build/html目录下的 `index.html`文件可以查看静态网页

   

