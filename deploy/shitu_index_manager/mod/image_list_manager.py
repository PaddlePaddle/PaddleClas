import os


class ImageListManager:
    """
    图像列表文件管理器
    """
    def __init__(self, file_path="", encoding="utf-8"):
        self.__filePath = ""
        self.__dirName = ""
        self.__dataList = {}
        self.__findLikeClassifyResult = []
        if file_path != "":
            self.readFile(file_path, encoding)

    @property
    def filePath(self):
        return self.__filePath

    @property
    def dirName(self):
        return self.__dirName

    @dirName.setter
    def dirName(self, value):
        self.__dirName = value

    @property
    def dataList(self):
        return self.__dataList

    @property
    def classifyList(self):
        return self.__dataList.keys()

    @property
    def findLikeClassifyResult(self):
        return self.__findLikeClassifyResult

    def imageList(self, classify: str):
        """
        获取分类下的图片列表

        Args:
            classify (str): 分类名称

        Returns:
            list: 图片列表
        """
        return self.__dataList[classify]

    def readFile(self, file_path: str, encoding="utf-8"):
        """
        读取文件内容

        Args:
            file_path (str): 文件路径
            encoding (str, optional): 文件编码. 默认 "utf-8".

        Raises:
            Exception: 文件不存在
        """
        if not os.path.exists(file_path):
            raise Exception("文件不存在：{}".format(file_path))
        self.__filePath = file_path
        self.__dirName = os.path.dirname(self.__filePath)
        self.__readData(file_path, encoding)

    def __readData(self, file_path: str, encoding="utf-8"):
        """
        读取文件内容

        Args:
            file_path (str): 文件路径
            encoding (str, optional): 文件编码. 默认 "utf-8".
        """
        with open(file_path, "r", encoding=encoding) as f:
            self.__dataList.clear()
            for line in f:
                line = line.rstrip("\n")
                data = line.split("\t")
                self.__appendData(data)

    def __appendData(self, data: list):
        """
        添加数据

        Args:
            data (list): 数据
        """
        if data[1] not in self.__dataList:
            self.__dataList[data[1]] = []
        self.__dataList[data[1]].append(data[0])

    def writeFile(self, file_path="", encoding="utf-8"):
        """
        写入文件

        Args:
            file_path (str, optional): 文件路径. 默认 "".
            encoding (str, optional): 文件编码. 默认 "utf-8".
        """
        if file_path == "":
            file_path = self.__filePath
        if not os.path.exists(file_path):
            return False
        self.__dirName = os.path.dirname(self.__filePath)
        lines = []
        for classify in self.__dataList.keys():
            for path in self.__dataList[classify]:
                lines.append("{}\t{}\n".format(path, classify))
        with open(file_path, "w", encoding=encoding) as f:
            f.writelines(lines)
        return True

    def realPath(self, image_path: str):
        """
        获取真实路径

        Args:
            image_path (str): 图片路径
        """
        return os.path.join(self.__dirName, image_path)

    def realPathList(self, classify: str):
        """
        获取分类下的真实路径列表

        Args:
            classify (str): 分类名称

        Returns:
            list: 真实路径列表
        """
        if classify not in self.classifyList:
            return []
        paths = self.__dataList[classify]
        if len(paths) == 0:
            return []
        for i in range(len(paths)):
            paths[i] = os.path.join(self.__dirName, paths[i])
        return paths

    def findLikeClassify(self, name: str):
        """
        查找类似的分类名称

        Args:
            name (str): 分类名称

        Returns:
            list: 类似的分类名称列表
        """
        self.__findLikeClassifyResult.clear()
        for classify in self.__dataList.keys():
            word = str(name)
            if (word in classify):
                self.__findLikeClassifyResult.append(classify)
        return self.__findLikeClassifyResult

    def addClassify(self, classify: str):
        """
        添加分类

        Args:
            classify (str): 分类名称

        Returns:
            bool: 如果分类名称已经存在，返回False，否则添加分类并返回True
        """
        if classify in self.__dataList:
            return False
        self.__dataList[classify] = []
        return True

    def removeClassify(self, classify: str):
        """
        移除分类

        Args:
            classify (str): 分类名称

        Returns:
            bool: 如果分类名称不存在，返回False，否则移除分类并返回True
        """
        if classify not in self.__dataList:
            return False
        self.__dataList.pop(classify)
        return True

    def renameClassify(self, old_classify: str, new_classify: str):
        """
        重命名分类名称

        Args:
            old_classify (str): 原分类名称
            new_classify (str): 新分类名称

        Returns:
            bool: 如果原分类名称不存在，或者新分类名称已经存在，返回False，否则重命名分类名称并返回True
        """
        if old_classify not in self.__dataList:
            return False
        if new_classify in self.__dataList:
            return False
        self.__dataList[new_classify] = self.__dataList[old_classify]
        self.__dataList.pop(old_classify)
        return True

    def allClassfiyNotEmpty(self):
        """
        检查所有分类是否都有图片

        Returns:
            bool: 如果有一个分类没有图片，返回False，否则返回True
        """
        for classify in self.__dataList.keys():
            if len(self.__dataList[classify]) == 0:
                return False
        return True

    def resetImageList(self, classify: str, image_list: list):
        """
        重置图片列表

        Args:
            classify (str): 分类名称
            image_list (list): 图片相对路径列表

        Returns:
            bool: 如果分类名称不存在，返回False，否则重置图片列表并返回True
        """
        if classify not in self.__dataList:
            return False
        self.__dataList[classify] = image_list
        return True
