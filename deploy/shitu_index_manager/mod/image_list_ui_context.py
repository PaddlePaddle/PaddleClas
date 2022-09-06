import os
from stat import filemode

from PyQt5 import QtCore, QtGui, QtWidgets
from mod import image_list_manager as imglistmgr
from mod import utils
from mod import ui_renameclassifydialog
from mod import imageeditclassifydialog

# 图像缩放基数
BASE_IMAGE_SIZE = 64


class ImageListUiContext(QtCore.QObject):
    # 图片列表界面相关业务，style sheet 在 MainWindow.ui 相应的 ImageListWidget 中设置
    listCount = QtCore.pyqtSignal(int)  # 图像列表图像的数量
    selectedCount = QtCore.pyqtSignal(int)  # 图像列表选择图像的数量

    def __init__(self, ui: QtWidgets.QListWidget,
                 parent: QtWidgets.QMainWindow,
                 image_list_mgr: imglistmgr.ImageListManager):
        super(ImageListUiContext, self).__init__()
        self.__ui = ui
        self.__parent = parent
        self.__imageListMgr = image_list_mgr
        self.__initUi()
        self.__menu = QtWidgets.QMenu()
        self.__initMenu()
        self.__connectSignal()
        self.__selectedClassify = ""
        self.__imageScale = 1

    @property
    def ui(self):
        return self.__ui

    @property
    def parent(self):
        return self.__parent

    @property
    def imageListManager(self):
        return self.__imageListMgr

    @property
    def menu(self):
        return self.__menu

    def __initUi(self):
        """初始化图片列表样式"""
        self.__ui.setViewMode(QtWidgets.QListView.IconMode)
        self.__ui.setSpacing(15)
        self.__ui.setMovement(QtWidgets.QListView.Static)
        self.__ui.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

    def __initMenu(self):
        """初始化图片列表界面菜单"""
        utils.setMenu(self.__menu, "添加图片", self.addImage)
        utils.setMenu(self.__menu, "移除图片", self.removeImage)
        utils.setMenu(self.__menu, "编辑图片分类", self.editImageClassify)
        self.__menu.addSeparator()
        utils.setMenu(self.__menu, "选择全部图片", self.selectAllImage)
        utils.setMenu(self.__menu, "反向选择图片", self.reverseSelectImage)
        utils.setMenu(self.__menu, "取消选择图片", self.cancelSelectImage)

        self.__ui.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.__ui.customContextMenuRequested.connect(self.__showMenu)

    def __showMenu(self, pos):
        """显示图片列表界面菜单"""
        if len(self.__imageListMgr.filePath) > 0:
            self.__menu.exec_(self.__ui.mapToGlobal(pos))

    def __connectSignal(self):
        """连接信号与槽"""
        self.__ui.itemSelectionChanged.connect(self.onSelectionChanged)

    def setImageScale(self, scale: int):
        """设置图片大小"""
        self.__imageScale = scale
        size = QtCore.QSize(scale * BASE_IMAGE_SIZE, scale * BASE_IMAGE_SIZE)
        self.__ui.setIconSize(size)
        for i in range(self.__ui.count()):
            item = self.__ui.item(i)
            item.setSizeHint(size)

    def setImageList(self, classify: str):
        """设置图片列表"""
        size = QtCore.QSize(self.__imageScale * BASE_IMAGE_SIZE,
                            self.__imageScale * BASE_IMAGE_SIZE)
        self.__selectedClassify = classify
        image_list = self.__imageListMgr.imageList(classify)
        self.__ui.clear()
        count = 0
        for i in image_list:
            item = QtWidgets.QListWidgetItem(self.__ui)
            item.setIcon(QtGui.QIcon(self.__imageListMgr.realPath(i)))
            item.setData(QtCore.Qt.UserRole, i)
            item.setSizeHint(size)
            self.__ui.addItem(item)
            count += 1
        self.listCount.emit(count)

    def clear(self):
        """清除图片列表"""
        self.__ui.clear()

    def addImage(self):
        """添加图片"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self.__parent, "提示",
                                              "请先打开正确的图像库")
            return
        filter = "图片 (*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG);;所有文件(*.*)"
        dlg = QtWidgets.QFileDialog(self.__parent)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)  # 多选文件
        dlg.setViewMode(QtWidgets.QFileDialog.Detail)  # 详细模式
        file_paths = dlg.getOpenFileNames(filter=filter)[0]
        if len(file_paths) == 0:
            return
        image_list_dir = self.__imageListMgr.dirName
        file_list = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            new_file = self.__copyToImagesDir(path)
            if new_file != "" and image_list_dir in new_file:
                # 去掉 image_list_dir 的路径和斜杠
                begin = len(image_list_dir) + 1
                file_list.append(new_file[begin:])
        if len(file_list) > 0:
            if self.__selectedClassify == "":
                QtWidgets.QMessageBox.warning(self.__parent, "提示", "请先选择分类")
                return
            new_list = self.__imageListMgr.imageList(
                self.__selectedClassify) + file_list
            self.__imageListMgr.resetImageList(self.__selectedClassify,
                                               new_list)
            self.setImageList(self.__selectedClassify)
            self.__imageListMgr.writeFile()

    def __copyToImagesDir(self, image_path: str):
        md5 = utils.fileMD5(image_path)
        file_ext = utils.fileExtension(image_path)
        to_dir = os.path.join(self.__imageListMgr.dirName, "images")
        new_path = os.path.join(to_dir, md5 + file_ext)
        if os.path.exists(to_dir):
            utils.copyFile(image_path, new_path)
            return new_path
        else:
            return ""

    def removeImage(self):
        """移除图片"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self.__parent, "提示",
                                              "请先打开正确的图像库")
            return
        path_list = []
        image_list = self.__ui.selectedItems()
        if len(image_list) == 0:
            return
        question = QtWidgets.QMessageBox.question(self.__parent, "移除图片",
                                                  "确定移除所选图片吗？")
        if question == QtWidgets.QMessageBox.No:
            return
        for i in range(self.__ui.count()):
            item = self.__ui.item(i)
            img_path = item.data(QtCore.Qt.UserRole)
            if not item.isSelected():
                path_list.append(img_path)
            else:
                # 从磁盘上删除图片
                utils.removeFile(
                    os.path.join(self.__imageListMgr.dirName, img_path))
        self.__imageListMgr.resetImageList(self.__selectedClassify, path_list)
        self.setImageList(self.__selectedClassify)
        self.__imageListMgr.writeFile()

    def editImageClassify(self):
        """编辑图片分类"""
        old_classify = self.__selectedClassify
        dlg = imageeditclassifydialog.ImageEditClassifyDialog(
            parent=self.__parent,
            old_classify=old_classify,
            classify_list=self.__imageListMgr.classifyList)
        result = dlg.exec_()
        new_classify = dlg.newClassify
        if result == QtWidgets.QDialog.Accepted \
                and new_classify != old_classify \
                and new_classify != "":
            self.__moveImage(old_classify, new_classify)
            self.__imageListMgr.writeFile()

    def __moveImage(self, old_classify, new_classify):
        """移动图片"""
        keep_list = []
        is_selected = False
        move_list = self.__imageListMgr.imageList(new_classify)
        for i in range(self.__ui.count()):
            item = self.__ui.item(i)
            txt = item.data(QtCore.Qt.UserRole)
            if item.isSelected():
                move_list.append(txt)
                is_selected = True
            else:
                keep_list.append(txt)
        if is_selected:
            self.__imageListMgr.resetImageList(new_classify, move_list)
            self.__imageListMgr.resetImageList(old_classify, keep_list)
            self.setImageList(old_classify)

    def selectAllImage(self):
        """选择所有图片"""
        self.__ui.selectAll()

    def reverseSelectImage(self):
        """反向选择图片"""
        for i in range(self.__ui.count()):
            item = self.__ui.item(i)
            item.setSelected(not item.isSelected())

    def cancelSelectImage(self):
        """取消选择图片"""
        self.__ui.clearSelection()

    def onSelectionChanged(self):
        """选择图像该变，发送选择的数量信号"""
        count = len(self.__ui.selectedItems())
        self.selectedCount.emit(count)
