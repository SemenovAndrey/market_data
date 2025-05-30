from PyQt5 import QtWidgets
import sys


if __name__ == '__main__':
    from client.ui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
