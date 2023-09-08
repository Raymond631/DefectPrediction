# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *

from childWindow import *
from xuezhang.model.dnn.mdp_dnn import nerual_network
from xuezhang.model.dnn.test_dnn import test_network, dnn_result
from xuezhang.model.knn.knn import test
from xuezhang.model.randm.mdp_random import random_forest
from xuezhang.model.randm.test_random import test_random, random_result


class Ui_Form(QWidget):

    def __init__(self):
        super(Ui_Form, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, Form):
        Form.setObjectName("软件缺陷预测系统")
        Form.resize(650, 450)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(30, 20, 600, 400))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 250, 120, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(40, 40, 171, 16))
        self.label.setObjectName("label")
        self.cf = QtWidgets.QPushButton(self.tab)
        self.cf.setGeometry(QtCore.QRect(200, 30, 93, 28))
        self.cf.setObjectName("cf")
        self.label_6 = QtWidgets.QLabel(self.tab)
        self.label_6.setGeometry(QtCore.QRect(300, 40, 350, 16))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(10, 30, 191, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(250, 30, 271, 16))
        self.label_12.setObjectName("label_12")
        self.cf2 = QtWidgets.QPushButton(self.tab_2)
        self.cf2.setGeometry(QtCore.QRect(250, 60, 93, 28))
        self.cf2.setObjectName("cf2")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(250, 100, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(40, 210, 111, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        self.label_14.setGeometry(QtCore.QRect(150, 210, 31, 16))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setGeometry(QtCore.QRect(40, 240, 121, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.tab_2)
        self.label_16.setGeometry(QtCore.QRect(160, 240, 31, 16))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setGeometry(QtCore.QRect(40, 200, 261, 16))
        self.label_17.setObjectName("label_17")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_5.setGeometry(QtCore.QRect(170, 190, 93, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.radioButton = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton.setGeometry(QtCore.QRect(30, 70, 115, 19))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton_2.setGeometry(QtCore.QRect(30, 110, 151, 19))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_3.setGeometry(QtCore.QRect(40, 110, 151, 19))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_4.setGeometry(QtCore.QRect(40, 150, 151, 19))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_5.setGeometry(QtCore.QRect(40, 190, 151, 19))
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_6 = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton_6.setGeometry(QtCore.QRect(30, 150, 151, 19))
        self.radioButton_6.setObjectName("radioButton_6")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(350, 65, 350, 16))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.tabWidget.addTab(self.tab_2, "")

        QtCore.QMetaObject.connectSlotsByName(Form)
        self.cf.clicked.connect(self.openfile)
        self.cf2.clicked.connect(self.openfile2)
        self.pushButton.clicked.connect(self.test)
        self.pushButton_2.clicked.connect(self.train)
        self.pushButton_5.clicked.connect(self.view_report)
        self.retranslateUi(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "软件缺陷预测系统"))
        self.pushButton_2.setText(_translate("Form", "模型训练"))
        self.label.setText(_translate("Form", "上传软件缺陷数据集："))
        self.cf.setText(_translate("Form", "选择文件"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "模型训练"))
        self.label_11.setText(_translate("Form", "选择预测模型："))
        self.radioButton.setText(_translate("Form", "随机森林模型"))
        self.radioButton_2.setText(_translate("Form", "深度神经网络模型"))
        self.radioButton_5.setText(_translate("Form", "k-邻近模型"))
        self.radioButton_3.setText(_translate("Form", "随机森林模型"))
        self.radioButton_4.setText(_translate("Form", "深度神经网络模型"))
        self.radioButton_6.setText(_translate("Form", "k-邻近模型"))
        self.label_12.setText(_translate("Form", "上传需要预测的软件缺陷集："))
        self.cf2.setText(_translate("Form", "选择文件"))
        self.pushButton.setText(_translate("Form", "开始预测"))

        self.label_17.setText(_translate("Form", "保存预测报告:"))
        self.pushButton_5.setText(_translate("Form", "保存"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "缺陷预测"))

    def train(self):
        if self.radioButton_3.isChecked():
            random_forest()
        elif self.radioButton_4.isChecked():
            nerual_network()
        elif self.radioButton_5.isChecked():
            test('MDP/training_mini.csv', 'MDP/test_mini.csv')

    def test(self):
        if self.radioButton.isChecked():
            random_result()
        elif self.radioButton_2.isChecked():
            dnn_result()
        elif self.radioButton_6.isChecked():
            test('MDP/training_mini.csv', 'MDP/test_mini.csv')

    def view_report(self):
        if self.radioButton.isChecked():
            test_random()
        elif self.radioButton_2.isChecked():
            test_network()
        elif self.radioButton_6.isChecked():
            test_network()

    def openfile(self):
        openfile_name, filetype = QFileDialog.getOpenFileName(self, '选择文件', "/", "All Files (*);;Excel files (*.xlsx , *.xls , *.csv)")
        self.label_6.setText(openfile_name)

    def openfile2(self):
        openfile_name, filetype = QFileDialog.getOpenFileName(self, '选择文件', "/", "All Files (*);;Excel files (*.xlsx , *.xls , *.csv)")
        self.label_7.setText(openfile_name)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # 实例化主窗口
    Widgets = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Widgets)

    # 实例化子窗口
    child = QDialog()
    child_ui = Ui_Dialog()
    child_ui.setupUi(child)

    # 按钮绑定事件
    btn = ui.pushButton_5
    btn.clicked.connect(child.show)
    Widgets.show()
    sys.exit(app.exec_())
