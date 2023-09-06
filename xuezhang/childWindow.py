from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_Dialog(object):
  def setupUi(self, Dialog):
    Dialog.setObjectName("Dialog")
    Dialog.resize(300, 200)
    self.label = QtWidgets.QLabel(Dialog)
    self.label.setGeometry(QtCore.QRect(90, 35, 300, 100))
    self.label.setText("   保存成功!\n\n请到文件夹中查看")
    Dialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)  #设置窗体总显示在最上面
    self.retranslateUi(Dialog)
    QtCore.QMetaObject.connectSlotsByName(Dialog)

  def retranslateUi(self, Dialog):
    _translate = QtCore.QCoreApplication.translate
    Dialog.setWindowTitle(_translate("Dialog", "提示"))
