from PyQt5.QtWidgets import QApplication
import sys
from main_controller import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())