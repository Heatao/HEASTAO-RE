import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QAction, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QLineEdit
from baseline_svm.get_relation_svm import svm_pre
import os


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 textbox'
        self.left = 20
        self.top = 20
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(40, 40)
        self.textbox.resize(560, 80)

        # Create a button in the window
        self.button = QPushButton('get triple', self)
        self.button.move(500, 160)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        pre_relation = svm_pre(textboxValue, utils_path='../baseline_svm/utils')

        # QMessageBox.question(self, "Message", 'Relation: ' + pre_relation[0],
        #                      QMessageBox.Ok, QMessageBox.Ok)
        QMessageBox.about(self, 'Triple', pre_relation[0])
        """打印完毕之后清空文本框"""
        self.textbox.setText('')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exit(app.exec_())
