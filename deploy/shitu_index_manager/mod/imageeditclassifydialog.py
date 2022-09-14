import os
from PyQt5 import QtCore, QtGui, QtWidgets
from mod import image_list_manager
from mod import ui_imageeditclassifydialog
from mod import utils


class ImageEditClassifyDialog(QtWidgets.QDialog):
    """图像编辑分类对话框"""
    def __init__(self, parent, old_classify, classify_list):
        super(ImageEditClassifyDialog, self).__init__(parent)
        self.ui = ui_imageeditclassifydialog.Ui_Dialog()
        self.ui.setupUi(self)  # 初始化主窗口界面
        self.__oldClassify = old_classify
        self.__classifyList = classify_list
        self.__newClassify = ""
        self.__searchResult = []
        self.__initUi()
        self.__connectSignal()

    @property
    def newClassify(self):
        return self.__newClassify

    def __initUi(self):
        self.ui.oldLineEdit.setText(self.__oldClassify)
        self.__setClassifyList(self.__classifyList)
        self.ui.classifyListView.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

    def __connectSignal(self):
        self.ui.classifyListView.clicked.connect(self.selectedListView)
        self.ui.searchButton.clicked.connect(self.searchClassify)

    def __setClassifyList(self, classify_list):
        list_model = QtCore.QStringListModel(classify_list)
        self.ui.classifyListView.setModel(list_model)

    def selectedListView(self, index):
        if not self.ui.classifyListView.currentIndex().isValid():
            return
        txt = index.data()
        self.ui.newLineEdit.setText(txt)
        self.__newClassify = txt

    def searchClassify(self):
        txt = self.ui.searchWordLineEdit.text()
        self.__searchResult.clear()
        for classify in self.__classifyList:
            if txt in classify:
                self.__searchResult.append(classify)
        self.__setClassifyList(self.__searchResult)
