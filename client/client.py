import base64
import json
import os
import sys
from typing import Container

import cv2
import numpy as np
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget
from win32 import win32api, win32gui, win32print
from win32.lib import win32con
from win32.win32api import GetSystemMetrics

#修改数据库连接，并进行数据库迁移操作


def get_real_resolution():
    """获取真实的分辨率"""
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self):
        hDC = win32gui.GetDC(0)
        # 横向分辨率
        w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
        # 纵向分辨率
        h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
        
        self.rate = (w,h)
        if self.rate != 1 and self.rate != 1.25 and self.rate != 1.5 and self.rate != 1.75 :
            self.rate = 1


        super(Ui_MainWindow, self).__init__()
  

        self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率

        self.cap = cv2.VideoCapture()       #视频流

        self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头

        self.set_ui()                       #初始化程序界面

        self.slot_init()                    #初始化槽函数

 
    '''程序界面布局'''

    def set_ui(self):
        self.resize(1080*self.rate, 720*self.rate)

        self.__layout_main = QtWidgets.QHBoxLayout()           #总布局

        self.__layout_fun_button = QtWidgets.QVBoxLayout()      #按键布局

        self.__layout_data_show = QtWidgets.QVBoxLayout()       #数据(视频)显示布局

      
        # 窗口无标题栏、窗口置顶、窗口透明
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)

        # 窗口图标
        self.icon = QIcon()
        self.icon.addPixmap(QPixmap("./image/icon.png"), QIcon.Normal, QIcon.On)
        self.setWindowIcon(self.icon)

        # 矩形框
        self.rect_label = QLabel(self)
        self.rect_label.setGeometry(QRect(0*self.rate, 0*self.rate, 1080*self.rate, 720*self.rate))
        self.rect_label.setStyleSheet("background-color: rgba(255, 255, 255, 0.7);"
                                      "border-width: 5px 5px 5px 5px;"
                                      "border:2px solid #5453FF;"
                                      "border-radius:0px;")
        
        # 打开相机按钮矩形框
        self.rect_label = QLabel(self)
        self.rect_label.setGeometry(QRect(920*self.rate, 225*self.rate, 95*self.rate, 60*self.rate))
        self.rect_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);"
                                      "border-width: 5px 5px 5px 5px;"
                                      "border:2px solid #5453FF;"
                                      "border-radius:15px;")

        # 打开相机按钮
        self.button_open_camera = QPushButton(self)
        self.button_open_camera.setGeometry(QRect(940*self.rate, 230*self.rate, 50*self.rate, 50*self.rate))
        self.button_open_camera.setStyleSheet("background: rgba(255, 255, 255, 0);"
                                        "color: #4E4EF2;"
                                        "font: 15pt \"华光粗圆_CNKI\";"
                                        "font-weight:bold;")
        self.button_open_camera.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_open_camera.setText("拍摄")


        # 退出按钮矩形框
        self.rect_label = QLabel(self)
        self.rect_label.setGeometry(QRect(920*self.rate, 455*self.rate, 95*self.rate, 60*self.rate))
        self.rect_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);"
                                      "border-width: 5px 5px 5px 5px;"
                                      "border:2px solid #5453FF;"
                                      "border-radius:15px;")


        # 退出按钮
        self.button_close = QPushButton(self)
        self.button_close.setGeometry(QRect(940*self.rate, 460*self.rate, 50*self.rate, 50*self.rate))
        self.button_close.setStyleSheet("background: rgba(255, 255, 255, 0);"
                                        "color: #4E4EF2;"
                                        "font: 15pt  \"华光粗圆_CNKI\";"
                                        "font-weight:bold;"
                                      )

        self.button_close.setCursor(QCursor(Qt.PointingHandCursor))
        self.button_close.setText("退出")

        self.title_label = QLabel(self)
        self.title_label.setGeometry(QRect(180*self.rate, 25*self.rate, 420*self.rate, 50*self.rate))
        self.title_label.setStyleSheet("color: #5453FF;"
                                       "background: transparent;"
                                       "font: 20pt \"华光粗圆_CNKI\";"
                                       "font-weight:bold;")
        self.title_label.setText("袋鼯麻麻——智能购物平台")

        self.setWindowTitle('袋鼯麻麻——智能购物平台')

        # Paddle Logo
        self.paddle_logo=QLabel("图片",self)
        self.paddle_pic=QPixmap("./image/paddlepaddle.png")
        self.paddle_logo.setPixmap(self.paddle_pic)
        self.paddle_logo.setGeometry(QRect(3*self.rate, 3*self.rate, 400*self.rate,105*self.rate))


        # 袋鼯 Logo
        self.daiwu_logo=QLabel("图片",self)
        self.daiwu_pic=QPixmap("./image/icon.png")
        self.daiwu_logo.setPixmap(self.daiwu_pic)
        self.daiwu_logo.setGeometry(QRect(881*self.rate, 10*self.rate, 190*self.rate, 150*self.rate))


        # 拍摄识别框
        self.rect_label = QLabel(self)

        self.rect_label.setGeometry(QRect(60*self.rate, 110*self.rate, 820*self.rate,560*self.rate))
        self.rect_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);"
                                      "border-width: 5px 5px 5px 5px;"
                                      "border:1px solid  #5453FF;"
                                      "border-radius:15px;"
                                      "font-weight:bold;")

        # add logo
        self.add_logo=QLabel("图片",self)
        self.add_pic=QPixmap("./image/add.png")
        self.add_logo.setPixmap(self.add_pic)
        self.add_logo.setGeometry(QRect(410*self.rate, 200*self.rate, 120*self.rate,120*self.rate))


        self.text_label = QLabel(self)
        self.text_label.setGeometry(QRect(200*self.rate, 350*self.rate, 540*self.rate, 300*self.rate))
        self.text_label.setStyleSheet("background-color: rgba(255, 255, 255, 0);"
                                      "border-width: 5px 5px 5px 5px;"
                                      "border:2px dotted #5453FF;"
                                      "border-radius:15px;")

        self.text = QLabel(self)
        self.text.setGeometry(QRect(340*self.rate, 350*self.rate, 420*self.rate, 50*self.rate))
        self.text.setStyleSheet("color: #5453FF;"
                                       "background: transparent;"
                                       "font: 10pt \"华光粗圆_CNKI\";"
                                       )


        self.text.setText("袋鼯麻麻——智能购物平台使用指南")
        self.note = QLabel(self)
        self.note.setGeometry(QRect(290*self.rate, 420*self.rate, 420*self.rate, 200*self.rate))
        self.note.setStyleSheet("color: #5453FF;"
                                       "background: transparent;"
                                       "font: 10pt \"华光粗圆_CNKI\";"
                                       )
        self.note.setText("->点击拍摄即可打开摄像头；\n\n->请务必将商品放置在摄像区域，以免识别遗漏；\n\n->放置完毕后点击识别等待片刻；\n\n->片刻后系统将自动结算商品总价；\n\n->若系统未正常返回商品价格，则证明\n    存在数据库中没有的商品，请联系管理员更新。")


        # 版本号
        self.version_label = QLabel(self)
        self.version_label.setGeometry(QRect(920*self.rate, 600*self.rate, 200*self.rate, 20*self.rate))
        self.version_label.setStyleSheet("color: #5453FF;"
                                         "background: transparent;"
                                         "font: 10pt \"微软雅黑\";"
                                         "font-weight:bold;")
        self.version_label.setText("版本号: Ver 1.0 ")


        self.author_label = QLabel(self)
        self.author_label.setGeometry(QRect(920*self.rate, 630*self.rate, 200*self.rate, 20*self.rate))
        self.author_label.setStyleSheet("color: #5453FF;"
                                         "background: transparent;"
                                         "font: 8pt \"微软雅黑\";"
                                         )
        self.author_label.setText("——By thomas-yanxin")


        # '''信息显示'''

        self.label_show_camera = QtWidgets.QLabel(self)   #定义显示视频的Label
        self.label_show_camera.setGeometry(QRect(60*self.rate, 110*self.rate, 820*self.rate,560*self.rate))
        self.label_show_camera.setFixedSize(820*self.rate,560*self.rate)    


    '''初始化所有槽函数'''

    def slot_init(self):

        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)    #若该按键被点击，则调用button_open_camera_clicked()

        self.timer_camera.timeout.connect(self.show_camera) #若定时器结束，则调用show_camera()

        self.button_close.clicked.connect(self.close)#若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序


    '''槽函数之一'''

    def button_open_camera_clicked(self):

        if self.timer_camera.isActive() == False:   #若定时器未启动

            flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

            if flag == False:       #flag表示open()成不成功

                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)

            else:

                self.timer_camera.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

                self.button_open_camera.setText('识别')

        else:

            self.container_recognition()
            
    
    def open_camera(self):

        flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

        if flag == False:       #flag表示open()成不成功

            msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)

        else:

            self.timer_camera.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示



    def lable_close(self):

        if self.timer_camera.isActive():

            self.timer_camera.stop()

        if self.cap.isOpened():

            self.cap.release()

        self.label_show_camera.clear()


    def show_camera(self):

        flag,self.image = self.cap.read()  #从视频流中读取

        show = cv2.resize(self.image,(820*self.rate,560*self.rate))     #把读到的帧的大小重新设置为 640x480

        show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色

        showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式

        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage


    def deplay(self, value):

        '''显示对话框返回值'''
        if value == "Please connect root to upload container's name and it's price!\n":
        
            QMessageBox.information(self, "Warning","{}".format(value), QMessageBox.Yes | QMessageBox.No)

        else:
            QMessageBox.information(self, "价格清单","您一共购买了：\n{}".format(value), QMessageBox.Yes | QMessageBox.No)


    def getByte(self, path):

        with open(path, 'rb') as f:

            img_byte = base64.b64encode(f.read()) #二进制读取后变base64编码

        img_str = img_byte.decode('ascii') #转成python的unicode

        return img_str 
    

    def container_recognition(self):

        self.picture_file = './test_client_pic.jpg'

        cv2.imwrite(self.picture_file, self.image)

        img_str = self.getByte('./test_client_pic.jpg')

        requestsss={'name':'测试图片', 'image':img_str}
        req = json.dumps(requestsss) #字典数据结构变json(所有程序语言都认识的字符串)

        res=requests.post('localhost/reference_client/', data=req)
        print(type(res.text))
        json_res = json.loads(res.text)
        print(json_res['container'])
        container_all = json_res['container']
        if container_all =="Please connect root to upload container's name and it's price!\n":
            rec_deplay_str_all = container_all
        else:
            price_all = json_res['price_all']
            rec_docs_price_all = []
            
            for i in range(len(container_all)):
                rec_docs_price = []
                if i%2 == 0:
                    container = container_all[i]
                    price = container_all[i+1]
                    rec_docs_price.append(container)
                    rec_docs_price.append(price)
                    rec_docs_price_all.append(rec_docs_price)

            rec_deplay_str = ''

            for rec_single in rec_docs_price_all:
                rec_name = rec_single[0]
                rec_price = rec_single[1]
                rec_deplay_str = '商品：{}'.format(rec_name) + '\t' + '单价：{}元'.format(str(rec_price)) + '\n' + rec_deplay_str
                rec_deplay_str_all = rec_deplay_str + '\n' + '您需付款：{}元'.format(str(price_all))

        self.deplay(rec_deplay_str_all)


if __name__ =='__main__':

    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用

    ui = Ui_MainWindow()                    #实例化Ui_MainWindow

    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的

    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过
