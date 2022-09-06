import os

from PyQt5 import QtCore, QtWidgets
from mod import image_list_manager as imglistmgr
from mod import utils
from mod import ui_addclassifydialog
from mod import ui_renameclassifydialog


class ClassifyUiContext(QtCore.QObject):
    # 分类界面相关业务
    selected = QtCore.pyqtSignal(str)  # 选择分类信号

    def __init__(self, ui: QtWidgets.QListView, parent: QtWidgets.QMainWindow,
                 image_list_mgr: imglistmgr.ImageListManager):
        super(ClassifyUiContext, self).__init__()
        self.__ui = ui
        self.__parent = parent
        self.__imageListMgr = image_list_mgr
        self.__menu = QtWidgets.QMenu()
        self.__initMenu()
        self.__initUi()
        self.__connectSignal()

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
        """初始化分类界面"""
        self.__ui.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def __connectSignal(self):
        """连接信号"""
        self.__ui.clicked.connect(self.uiClicked)
        self.__ui.doubleClicked.connect(self.uiDoubleClicked)

    def __initMenu(self):
        """初始化分类界面菜单"""
        utils.setMenu(self.__menu, "添加分类", self.addClassify)
        utils.setMenu(self.__menu, "移除分类", self.removeClassify)
        utils.setMenu(self.__menu, "重命名分类", self.renemeClassify)

        self.__ui.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.__ui.customContextMenuRequested.connect(self.__showMenu)

    def __showMenu(self, pos):
        """显示分类界面菜单"""
        if len(self.__imageListMgr.filePath) > 0:
            self.__menu.exec_(self.__ui.mapToGlobal(pos))

    def setClassifyList(self, classify_list):
        """设置分类列表"""
        list_model = QtCore.QStringListModel(classify_list)
        self.__ui.setModel(list_model)

    def uiClicked(self, index):
        """分类列表点击"""
        if not self.__ui.currentIndex().isValid():
            return
        txt = index.data()
        self.selected.emit(txt)

    def uiDoubleClicked(self, index):
        """分类列表双击"""
        if not self.__ui.currentIndex().isValid():
            return
        ole_name = index.data()
        dlg = QtWidgets.QDialog(parent=self.parent)
        ui = ui_renameclassifydialog.Ui_RenameClassifyDialog()
        ui.setupUi(dlg)
        ui.oldNameLineEdit.setText(ole_name)
        result = dlg.exec_()
        new_name = ui.newNameLineEdit.text()
        if result == QtWidgets.QDialog.Accepted:
            mgr_result = self.__imageListMgr.renameClassify(ole_name, new_name)
            if not mgr_result:
                QtWidgets.QMessageBox.warning(self.parent, "重命名分类", "重命名分类错误")
            else:
                self.setClassifyList(self.__imageListMgr.classifyList)
                self.__imageListMgr.writeFile()

    def addClassify(self):
        """添加分类"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self.__parent, "提示",
                                              "请先打开正确的图像库")
            return
        dlg = QtWidgets.QDialog(parent=self.parent)
        ui = ui_addclassifydialog.Ui_AddClassifyDialog()
        ui.setupUi(dlg)
        result = dlg.exec_()
        txt = ui.lineEdit.text()
        if result == QtWidgets.QDialog.Accepted:
            mgr_result = self.__imageListMgr.addClassify(txt)
            if not mgr_result:
                QtWidgets.QMessageBox.warning(self.parent, "添加分类", "添加分类错误")
            else:
                self.setClassifyList(self.__imageListMgr.classifyList)

    def removeClassify(self):
        """移除分类"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self.__parent, "提示",
                                              "请先打开正确的图像库")
            return
        if not self.__ui.currentIndex().isValid():
            return
        classify = self.__ui.currentIndex().data()
        result = QtWidgets.QMessageBox.information(
            self.parent,
            "移除分类",
            "确定移除分类: {}".format(classify),
            buttons=QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
            defaultButton=QtWidgets.QMessageBox.Cancel)
        if result == QtWidgets.QMessageBox.Ok:
            if len(self.__imageListMgr.imageList(classify)) > 0:
                QtWidgets.QMessageBox.warning(self.parent, "移除分类",
                                              "分类下存在图片，请先移除图片")
            else:
                self.__imageListMgr.removeClassify(classify)
                self.setClassifyList(self.__imageListMgr.classifyList())

    def renemeClassify(self):
        """重命名分类"""
        idx = self.__ui.currentIndex()
        if idx.isValid():
            self.uiDoubleClicked(idx)

    def searchClassify(self, classify):
        """查找分类"""
        self.setClassifyList(self.__imageListMgr.findLikeClassify(classify))
