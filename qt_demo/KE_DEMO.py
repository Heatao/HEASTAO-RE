import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QTextEdit
from baseline_svm.get_relation_svm import svm_pre
from NER.nltk_ner import get_entities


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Knowledge Extraction'
        self.left = 20
        self.top = 20
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create textboxs
        self.textbox1 = QTextEdit(self)
        self.textbox1.move(40, 40)
        self.textbox1.resize(560, 80)
        self.textbox1.setPlainText('The company fabricates plastic chairs.')    # set an example

        self.textbox2 = QTextEdit(self)
        self.textbox2.move(40, 230)
        self.textbox2.resize(560, 80)

        # Create a button in the window
        self.button1 = QPushButton('Get triple', self)
        self.button1.move(500, 160)

        # connect button to function on_click
        self.button1.clicked.connect(self.on_click)

        # set button2 to clear textbox1
        self.button2 = QPushButton('Clear', self)
        self.button2.move(420, 160)
        self.button2.clicked.connect(self.clear_click)

        # show must be the last
        self.show()

    @pyqtSlot()
    def on_click(self):
        self.textbox2.clear()
        textboxValue = self.textbox1.toPlainText()
        pre_relation = svm_pre(textboxValue, utils_path='../baseline_svm/utils')
        entities = get_entities(textboxValue)
        triple = entities[0] + ' - ' + pre_relation[0] + ' - ' + entities[1]
        self.textbox2.setPlainText(triple)

    @pyqtSlot()
    def clear_click(self):
        self.textbox1.clear()
        self.textbox2.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exit(app.exec_())
