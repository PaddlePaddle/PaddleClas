from multiprocessing.dummy import active_children
from multiprocessing import Process
import os
import sys
import socket

from PyQt5 import QtCore, QtGui, QtWidgets
from mod import ui_mainwindow
from mod import image_list_manager
from mod import classify_ui_context
from mod import image_list_ui_context
from mod import ui_newlibrarydialog
from mod import index_http_client
from mod import utils
from mod import ui_waitdialog
import threading

TOOL_BTN_ICON_SIZE = 64
TOOL_BTN_ICON_SMALL = 48

try:
    DEFAULT_HOST = socket.gethostbyname(socket.gethostname())
except:
    DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000
PADDLECLAS_DOC_URL = "https://gitee.com/paddlepaddle/PaddleClas/docs/zh_CN/inference_deployment/shitu_gallery_manager.md"


class MainWindow(QtWidgets.QMainWindow):
    """主窗口"""
    newIndexMsg = QtCore.pyqtSignal(str)  # 新建索引库线程信号
    openIndexMsg = QtCore.pyqtSignal(str)  # 打开索引库线程信号
    updateIndexMsg = QtCore.pyqtSignal(str)  # 更新索引库线程信号
    importImageCount = QtCore.pyqtSignal(int)  # 导入图像数量信号

    def __init__(self, ip=None, port=None):
        super(MainWindow, self).__init__()
        if ip is not None and port is not None:
            self.server_ip = ip
            self.server_port = port
        else:
            self.server_ip = DEFAULT_HOST
            self.server_port = DEFAULT_PORT

        self.ui = ui_mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)  # 初始化主窗口界面
        self.__imageListMgr = image_list_manager.ImageListManager()

        self.__appMenu = QtWidgets.QMenu()  # 应用菜单
        self.__libraryAppendMenu = QtWidgets.QMenu()  # 图像库附加功能菜单
        self.__initAppMenu()  # 初始化应用菜单

        self.__pathBar = QtWidgets.QLabel(self)  # 路径
        self.__classifyCountBar = QtWidgets.QLabel(self)  # 分类数量
        self.__imageCountBar = QtWidgets.QLabel(self)  # 图像列表数量
        self.__imageSelectedBar = QtWidgets.QLabel(self)  # 图像列表选择数量
        self.__spaceBar1 = QtWidgets.QLabel(self)  # 空格间隔栏
        self.__spaceBar2 = QtWidgets.QLabel(self)  # 空格间隔栏
        self.__spaceBar3 = QtWidgets.QLabel(self)  # 空格间隔栏

        # 分类界面相关业务
        self.__classifyUiContext = classify_ui_context.ClassifyUiContext(
            ui=self.ui.classifyListView,
            parent=self,
            image_list_mgr=self.__imageListMgr)

        # 图片列表界面相关业务
        self.__imageListUiContext = image_list_ui_context.ImageListUiContext(
            ui=self.ui.imageListWidget,
            parent=self,
            image_list_mgr=self.__imageListMgr)

        # 搜索的历史记录回车快捷键
        self.__historyCmbShortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Return),
            self.ui.searchClassifyHistoryCmb)

        self.__waitDialog = QtWidgets.QDialog()  # 等待对话框
        self.__waitDialogUi = ui_waitdialog.Ui_WaitDialog()  # 等待对话框界面
        self.__initToolBtn()
        self.__connectSignal()
        self.__initUI()
        self.__initWaitDialog()

    def __initUI(self):
        """初始化界面"""
        # 窗口图标
        self.setWindowIcon(QtGui.QIcon("./resource/app_icon.png"))

        # 初始化分割窗口
        self.ui.splitter.setStretchFactor(0, 20)
        self.ui.splitter.setStretchFactor(1, 80)

        # 初始化图像缩放
        self.ui.imageScaleSlider.setValue(4)

        # 状态栏界面设置
        space_bar = "                "  # 间隔16空格
        self.__spaceBar1.setText(space_bar)
        self.__spaceBar2.setText(space_bar)
        self.__spaceBar3.setText(space_bar)
        self.ui.statusbar.addWidget(self.__pathBar)
        self.ui.statusbar.addWidget(self.__spaceBar1)
        self.ui.statusbar.addWidget(self.__classifyCountBar)
        self.ui.statusbar.addWidget(self.__spaceBar2)
        self.ui.statusbar.addWidget(self.__imageCountBar)
        self.ui.statusbar.addWidget(self.__spaceBar3)
        self.ui.statusbar.addWidget(self.__imageSelectedBar)

    def __initToolBtn(self):
        """初始化工具按钮"""
        self.__setToolButton(self.ui.appMenuBtn, "应用菜单",
                             "./resource/app_menu.png", TOOL_BTN_ICON_SIZE)

        self.__setToolButton(self.ui.saveImageLibraryBtn, "保存图像库",
                             "./resource/save_image_Library.png",
                             TOOL_BTN_ICON_SIZE)
        self.ui.saveImageLibraryBtn.clicked.connect(self.saveImageLibrary)

        self.__setToolButton(self.ui.addClassifyBtn, "添加分类",
                             "./resource/add_classify.png", TOOL_BTN_ICON_SIZE)
        self.ui.addClassifyBtn.clicked.connect(
            self.__classifyUiContext.addClassify)

        self.__setToolButton(self.ui.removeClassifyBtn, "移除分类",
                             "./resource/remove_classify.png",
                             TOOL_BTN_ICON_SIZE)
        self.ui.removeClassifyBtn.clicked.connect(
            self.__classifyUiContext.removeClassify)

        self.__setToolButton(self.ui.searchClassifyBtn, "查找分类",
                             "./resource/search_classify.png",
                             TOOL_BTN_ICON_SMALL)
        self.ui.searchClassifyBtn.clicked.connect(
            self.__classifyUiContext.searchClassify)

        self.__setToolButton(self.ui.addImageBtn, "添加图片",
                             "./resource/add_image.png", TOOL_BTN_ICON_SMALL)
        self.ui.addImageBtn.clicked.connect(self.__imageListUiContext.addImage)

        self.__setToolButton(self.ui.removeImageBtn, "移除图片",
                             "./resource/remove_image.png",
                             TOOL_BTN_ICON_SMALL)
        self.ui.removeImageBtn.clicked.connect(
            self.__imageListUiContext.removeImage)

        self.ui.searchClassifyHistoryCmb.setToolTip("查找分类历史")
        self.ui.imageScaleSlider.setToolTip("图片缩放")

    def __setToolButton(self,
                        button,
                        tool_tip: str,
                        icon_path: str,
                        icon_size: int):
        """设置工具按钮"""
        button.setToolTip(tool_tip)
        button.setIcon(QtGui.QIcon(icon_path))
        button.setIconSize(QtCore.QSize(icon_size, icon_size))

    def __initAppMenu(self):
        """初始化应用菜单"""
        utils.setMenu(self.__appMenu, "新建图像库", self.newImageLibrary)
        utils.setMenu(self.__appMenu, "打开图像库", self.openImageLibrary)
        utils.setMenu(self.__appMenu, "保存图像库", self.saveImageLibrary)

        self.__libraryAppendMenu.setTitle("导入图像")
        utils.setMenu(self.__libraryAppendMenu, "导入 image_list 图像",
                      self.importImageListImage)
        utils.setMenu(self.__libraryAppendMenu, "导入多文件夹图像",
                      self.importDirsImage)
        self.__appMenu.addMenu(self.__libraryAppendMenu)

        self.__appMenu.addSeparator()
        utils.setMenu(self.__appMenu, "新建/重建 索引库", self.newIndexLibrary)
        utils.setMenu(self.__appMenu, "更新索引库", self.updateIndexLibrary)
        self.__appMenu.addSeparator()
        utils.setMenu(self.__appMenu, "帮助", self.showHelp)
        utils.setMenu(self.__appMenu, "关于", self.showAbout)
        utils.setMenu(self.__appMenu, "退出", self.exitApp)

        self.ui.appMenuBtn.setMenu(self.__appMenu)
        self.ui.appMenuBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

    def __initWaitDialog(self):
        """初始化等待对话框"""
        self.__waitDialogUi.setupUi(self.__waitDialog)
        self.__waitDialog.setWindowFlags(QtCore.Qt.Dialog |
                                         QtCore.Qt.FramelessWindowHint)

    def __startWait(self, msg: str):
        """开始显示等待对话框"""
        self.setEnabled(False)
        self.__waitDialogUi.msgLabel.setText(msg)
        self.__waitDialog.setWindowFlags(QtCore.Qt.Dialog |
                                         QtCore.Qt.FramelessWindowHint |
                                         QtCore.Qt.WindowStaysOnTopHint)
        self.__waitDialog.show()
        self.__waitDialog.repaint()

    def __stopWait(self):
        """停止显示等待对话框"""
        self.setEnabled(True)
        self.__waitDialogUi.msgLabel.setText("执行完毕！")
        self.__waitDialog.setWindowFlags(QtCore.Qt.Dialog |
                                         QtCore.Qt.FramelessWindowHint |
                                         QtCore.Qt.CustomizeWindowHint)
        self.__waitDialog.close()

    def __connectSignal(self):
        """连接信号与槽"""
        self.__classifyUiContext.selected.connect(
            self.__imageListUiContext.setImageList)
        self.ui.searchClassifyBtn.clicked.connect(self.searchClassify)
        self.ui.imageScaleSlider.valueChanged.connect(
            self.__imageListUiContext.setImageScale)
        self.__imageListUiContext.listCount.connect(self.__setImageCountBar)
        self.__imageListUiContext.selectedCount.connect(
            self.__setImageSelectedCountBar)
        self.__historyCmbShortcut.activated.connect(self.searchClassify)
        self.newIndexMsg.connect(self.__onNewIndexMsg)
        self.openIndexMsg.connect(self.__onOpenIndexMsg)
        self.updateIndexMsg.connect(self.__onUpdateIndexMsg)
        self.importImageCount.connect(self.__onImportImageCount)

    def newImageLibrary(self):
        """新建图像库"""
        dir_path = self.__openDirDialog("新建图像库")
        if dir_path == None:
            return
        if not utils.isEmptyDir(dir_path):
            QtWidgets.QMessageBox.warning(self, "错误", "该目录不为空，请选择空目录")
            return
        if not utils.initLibrary(dir_path):
            QtWidgets.QMessageBox.warning(self, "错误", "新建图像库失败")
            return
        QtWidgets.QMessageBox.information(self, "提示", "新建图像库成功")
        self.__reload(os.path.join(dir_path, "image_list.txt"), dir_path)

    def __openDirDialog(self, title: str):
        """打开目录对话框"""
        dlg = QtWidgets.QFileDialog(self)
        dlg.setWindowTitle(title)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            dir_path = dlg.selectedFiles()[0]
            return dir_path
        return None

    def openImageLibrary(self):
        """打开图像库"""
        dir_path = self.__openDirDialog("打开图像库")
        if dir_path != None:
            image_list_path = os.path.join(dir_path, "image_list.txt")
            if os.path.exists(image_list_path) \
                and os.path.exists(os.path.join(dir_path, "images")):
                self.__reload(image_list_path, dir_path)
                self.openIndexLibrary()

    def __reload(self, image_list_path: str, msg: str):
        """重新加载图像库"""
        self.__imageListMgr.readFile(image_list_path)
        self.__imageListUiContext.clear()
        self.__classifyUiContext.setClassifyList(
            self.__imageListMgr.classifyList)
        self.__setPathBar(msg)
        self.__setClassifyCountBar(len(self.__imageListMgr.classifyList))
        self.__setImageCountBar(0)
        self.__setImageSelectedCountBar(0)

    def saveImageLibrary(self):
        """保存图像库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.warning(self, "错误", "请先打开正确的图像库")
            return
        self.__imageListMgr.writeFile()
        self.__reload(self.__imageListMgr.filePath,
                      self.__imageListMgr.dirName)
        hint_str = "为保证图片准确识别，请在修改图片库后更新索引库。\n\
如果是新建图像库或者没有索引库，请新建索引库。"

        QtWidgets.QMessageBox.information(self, "提示", hint_str)

    def __onImportImageCount(self, count: int):
        """导入图像槽"""
        self.__stopWait()
        if count == -1:
            QtWidgets.QMessageBox.warning(self, "错误", "导入到当前图像库错误")
            return
        QtWidgets.QMessageBox.information(self, "提示",
                                          "导入图像库成功，导入图像：{}".format(count))
        self.__reload(self.__imageListMgr.filePath,
                      self.__imageListMgr.dirName)

    def __importImageListImageThread(self, from_path: str, to_path: str):
        """导入 image_list 图像 线程"""
        count = utils.oneKeyImportFromFile(
            from_path=from_path, to_path=to_path)
        if count == None:
            count = -1
        self.importImageCount.emit(count)

    def importImageListImage(self):
        """导入 image_list 图像 到当前图像库，建议当前库是新建的空库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self, "提示", "请先打开正确的图像库")
            return
        from_path = QtWidgets.QFileDialog.getOpenFileName(
            caption="导入 image_list 图像", filter="txt (*.txt)")[0]
        if not os.path.exists(from_path):
            QtWidgets.QMessageBox.information(self, "提示", "打开的文件不存在")
            return
        from_mgr = image_list_manager.ImageListManager(from_path)
        self.__startWait("正在导入图像，请等待。。。")
        thread = threading.Thread(
            target=self.__importImageListImageThread,
            args=(from_mgr.filePath, self.__imageListMgr.filePath))
        thread.start()

    def __importDirsImageThread(self, from_dir: str, to_image_list_path: str):
        """导入多文件夹图像 线程"""
        count = utils.oneKeyImportFromDirs(
            from_dir=from_dir, to_image_list_path=to_image_list_path)
        if count == None:
            count = -1
        self.importImageCount.emit(count)

    def importDirsImage(self):
        """导入 多文件夹图像 到当前图像库，建议当前库是新建的空库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self, "提示", "请先打开正确的图像库")
            return
        dir_path = self.__openDirDialog("导入多文件夹图像")
        if dir_path == None:
            return
        if not os.path.exists(dir_path):
            QtWidgets.QMessageBox.information(self, "提示", "打开的目录不存在")
            return
        self.__startWait("正在导入图像，请等待。。。")
        thread = threading.Thread(
            target=self.__importDirsImageThread,
            args=(dir_path, self.__imageListMgr.filePath))
        thread.start()

    def __newIndexThread(self,
                         index_root_path: str,
                         image_list_path: str,
                         index_method: str,
                         force: bool):
        """新建重建索引库线程"""
        try:
            client = index_http_client.IndexHttpClient(self.server_ip,
                                                       self.server_port)
            err_msg = client.new_index(
                image_list_path=image_list_path,
                index_root_path=index_root_path,
                index_method=index_method,
                force=force)
            if err_msg == None:
                err_msg = ""
            self.newIndexMsg.emit(err_msg)
        except Exception as e:
            self.newIndexMsg.emit(str(e))

    def __onNewIndexMsg(self, err_msg):
        """新建重建索引库槽"""
        self.__stopWait()
        if err_msg == "":
            QtWidgets.QMessageBox.information(self, "提示", "新建/重建 索引库成功")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", err_msg)

    def newIndexLibrary(self):
        """新建重建索引库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self, "提示", "请先打开正确的图像库")
            return
        dlg = QtWidgets.QDialog(self)
        ui = ui_newlibrarydialog.Ui_NewlibraryDialog()
        ui.setupUi(dlg)
        result = dlg.exec_()
        index_method = ui.indexMethodCmb.currentText()
        force = ui.resetCheckBox.isChecked()
        if result == QtWidgets.QDialog.Accepted:
            self.__startWait("正在 新建/重建 索引库，请等待。。。")
            thread = threading.Thread(
                target=self.__newIndexThread,
                args=(self.__imageListMgr.dirName, "image_list.txt",
                      index_method, force))
            thread.start()

    def __openIndexThread(self, index_root_path: str, image_list_path: str):
        """打开索引库线程"""
        try:
            client = index_http_client.IndexHttpClient(self.server_ip,
                                                       self.server_port)
            err_msg = client.open_index(
                index_root_path=index_root_path,
                image_list_path=image_list_path)
            if err_msg == None:
                err_msg = ""
            self.openIndexMsg.emit(err_msg)
        except Exception as e:
            self.openIndexMsg.emit(str(e))

    def __onOpenIndexMsg(self, err_msg):
        """打开索引库槽"""
        self.__stopWait()
        if err_msg == "":
            QtWidgets.QMessageBox.information(self, "提示", "打开索引库成功")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", err_msg)

    def openIndexLibrary(self):
        """打开索引库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self, "提示", "请先打开正确的图像库")
            return
        self.__startWait("正在打开索引库，请等待。。。")
        thread = threading.Thread(
            target=self.__openIndexThread,
            args=(self.__imageListMgr.dirName, "image_list.txt"))
        thread.start()

    def __updateIndexThread(self, index_root_path: str, image_list_path: str):
        """更新索引库线程"""
        try:
            client = index_http_client.IndexHttpClient(self.server_ip,
                                                       self.server_port)
            err_msg = client.update_index(
                image_list_path=image_list_path,
                index_root_path=index_root_path)
            if err_msg == None:
                err_msg = ""
            self.updateIndexMsg.emit(err_msg)
        except Exception as e:
            self.updateIndexMsg.emit(str(e))

    def __onUpdateIndexMsg(self, err_msg):
        """更新索引库槽"""
        self.__stopWait()
        if err_msg == "":
            QtWidgets.QMessageBox.information(self, "提示", "更新索引库成功")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", err_msg)

    def updateIndexLibrary(self):
        """更新索引库"""
        if not os.path.exists(self.__imageListMgr.filePath):
            QtWidgets.QMessageBox.information(self, "提示", "请先打开正确的图像库")
            return
        self.__startWait("正在更新索引库，请等待。。。")
        thread = threading.Thread(
            target=self.__updateIndexThread,
            args=(self.__imageListMgr.dirName, "image_list.txt"))
        thread.start()

    def searchClassify(self):
        """查找分类"""
        if len(self.__imageListMgr.classifyList) == 0:
            return
        cmb = self.ui.searchClassifyHistoryCmb
        txt = cmb.currentText()
        is_has = False
        if txt != "":
            for i in range(cmb.count()):
                if cmb.itemText(i) == txt:
                    is_has = True
                    break
            if not is_has:
                cmb.addItem(txt)
        self.__classifyUiContext.searchClassify(txt)

    def showHelp(self):
        """显示帮助"""
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(PADDLECLAS_DOC_URL))

    def showAbout(self):
        """显示关于对话框"""
        QtWidgets.QMessageBox.information(self, "关于", "识图图像库管理 V1.0.0")

    def exitApp(self):
        """退出应用"""
        sys.exit(0)

    def __setPathBar(self, msg: str):
        """设置路径状态栏信息"""
        self.__pathBar.setText("图像库路径：{}".format(msg))

    def __setClassifyCountBar(self, msg: str):
        self.__classifyCountBar.setText("分类总数量：{}".format(msg))

    def __setImageCountBar(self, count: int):
        """设置图像数量状态栏信息"""
        self.__imageCountBar.setText("当前图像数量：{}".format(count))

    def __setImageSelectedCountBar(self, count: int):
        """设置选择图像数量状态栏信息"""
        self.__imageSelectedBar.setText("选择图像数量：{}".format(count))
