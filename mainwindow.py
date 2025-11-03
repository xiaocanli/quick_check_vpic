# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMenu, QMenuBar, QPushButton, QScrollBar,
    QSizePolicy, QSlider, QSpacerItem, QSpinBox,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1600, 1080)
        self.actionClose_2 = QAction(MainWindow)
        self.actionClose_2.setObjectName(u"actionClose_2")
        self.actionCtrl_W = QAction(MainWindow)
        self.actionCtrl_W.setObjectName(u"actionCtrl_W")
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(31, 21, 1543, 325))
        self.horizontalLayout_32 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setEnabled(True)
        font = QFont()
        font.setFamilies([u"Helvetica"])
        font.setPointSize(12)
        font.setBold(False)
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.label_3)

        self.filetype_comboBox = QComboBox(self.layoutWidget)
        self.filetype_comboBox.addItem("")
        self.filetype_comboBox.addItem("")
        self.filetype_comboBox.setObjectName(u"filetype_comboBox")
        self.filetype_comboBox.setFont(font)

        self.horizontalLayout_4.addWidget(self.filetype_comboBox)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.plottype_comboBox = QComboBox(self.layoutWidget)
        self.plottype_comboBox.addItem("")
        self.plottype_comboBox.addItem("")
        self.plottype_comboBox.addItem("")
        self.plottype_comboBox.addItem("")
        self.plottype_comboBox.addItem("")
        self.plottype_comboBox.setObjectName(u"plottype_comboBox")
        self.plottype_comboBox.setFont(font)

        self.verticalLayout.addWidget(self.plottype_comboBox)

        self.integrate_checkBox = QCheckBox(self.layoutWidget)
        self.integrate_checkBox.setObjectName(u"integrate_checkBox")

        self.verticalLayout.addWidget(self.integrate_checkBox)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.layoutWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setEnabled(True)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.label_2)

        self.rawplot_comboBox = QComboBox(self.layoutWidget)
        self.rawplot_comboBox.setObjectName(u"rawplot_comboBox")
        self.rawplot_comboBox.setFont(font)

        self.horizontalLayout_3.addWidget(self.rawplot_comboBox)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_5 = QLabel(self.layoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_5)

        self.diagplot_comboBox = QComboBox(self.layoutWidget)
        self.diagplot_comboBox.setObjectName(u"diagplot_comboBox")
        self.diagplot_comboBox.setFont(font)

        self.horizontalLayout_2.addWidget(self.diagplot_comboBox)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_6 = QLabel(self.layoutWidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font)
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout.addWidget(self.label_6)

        self.overplot_comboBox = QComboBox(self.layoutWidget)
        self.overplot_comboBox.setObjectName(u"overplot_comboBox")
        self.overplot_comboBox.setFont(font)

        self.horizontalLayout.addWidget(self.overplot_comboBox)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.autoplot_checkBox = QCheckBox(self.layoutWidget)
        self.autoplot_checkBox.setObjectName(u"autoplot_checkBox")
        self.autoplot_checkBox.setFont(font)
        self.autoplot_checkBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.verticalLayout.addWidget(self.autoplot_checkBox)

        self.plotButton = QPushButton(self.layoutWidget)
        self.plotButton.setObjectName(u"plotButton")
        self.plotButton.setMinimumSize(QSize(100, 40))
        font1 = QFont()
        font1.setFamilies([u"Helvetica"])
        font1.setPointSize(16)
        font1.setBold(True)
        self.plotButton.setFont(font1)

        self.verticalLayout.addWidget(self.plotButton)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout.addItem(self.verticalSpacer_5)


        self.horizontalLayout_32.addLayout(self.verticalLayout)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(70, 16777215))
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_5.addWidget(self.label)

        self.xmin_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.xmin_SpinBox.setObjectName(u"xmin_SpinBox")
        self.xmin_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.xmin_SpinBox.setFont(font)
        self.xmin_SpinBox.setMinimum(-10.000000000000000)
        self.xmin_SpinBox.setMaximum(10.000000000000000)
        self.xmin_SpinBox.setValue(-10.000000000000000)

        self.horizontalLayout_5.addWidget(self.xmin_SpinBox)

        self.xmin_hScrollBar = QScrollBar(self.layoutWidget)
        self.xmin_hScrollBar.setObjectName(u"xmin_hScrollBar")
        self.xmin_hScrollBar.setMinimumSize(QSize(150, 0))
        self.xmin_hScrollBar.setFont(font)
        self.xmin_hScrollBar.setMinimum(-10)
        self.xmin_hScrollBar.setMaximum(10)
        self.xmin_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_5.addWidget(self.xmin_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_7 = QLabel(self.layoutWidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMaximumSize(QSize(70, 16777215))
        self.label_7.setFont(font)
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_6.addWidget(self.label_7)

        self.xmax_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.xmax_SpinBox.setObjectName(u"xmax_SpinBox")
        self.xmax_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.xmax_SpinBox.setFont(font)
        self.xmax_SpinBox.setMinimum(-10.000000000000000)
        self.xmax_SpinBox.setMaximum(10.000000000000000)
        self.xmax_SpinBox.setValue(10.000000000000000)

        self.horizontalLayout_6.addWidget(self.xmax_SpinBox)

        self.xmax_hScrollBar = QScrollBar(self.layoutWidget)
        self.xmax_hScrollBar.setObjectName(u"xmax_hScrollBar")
        self.xmax_hScrollBar.setMinimumSize(QSize(150, 0))
        self.xmax_hScrollBar.setFont(font)
        self.xmax_hScrollBar.setMinimum(-10)
        self.xmax_hScrollBar.setMaximum(10)
        self.xmax_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_6.addWidget(self.xmax_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_8 = QLabel(self.layoutWidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMaximumSize(QSize(70, 16777215))
        self.label_8.setFont(font)
        self.label_8.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_7.addWidget(self.label_8)

        self.ymin_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.ymin_SpinBox.setObjectName(u"ymin_SpinBox")
        self.ymin_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.ymin_SpinBox.setFont(font)
        self.ymin_SpinBox.setMinimum(-10.000000000000000)
        self.ymin_SpinBox.setMaximum(10.000000000000000)
        self.ymin_SpinBox.setValue(-10.000000000000000)

        self.horizontalLayout_7.addWidget(self.ymin_SpinBox)

        self.ymin_hScrollBar = QScrollBar(self.layoutWidget)
        self.ymin_hScrollBar.setObjectName(u"ymin_hScrollBar")
        self.ymin_hScrollBar.setMinimumSize(QSize(150, 0))
        self.ymin_hScrollBar.setFont(font)
        self.ymin_hScrollBar.setMinimum(-10)
        self.ymin_hScrollBar.setMaximum(10)
        self.ymin_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_7.addWidget(self.ymin_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_9 = QLabel(self.layoutWidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMaximumSize(QSize(70, 16777215))
        self.label_9.setFont(font)
        self.label_9.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_8.addWidget(self.label_9)

        self.ymax_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.ymax_SpinBox.setObjectName(u"ymax_SpinBox")
        self.ymax_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.ymax_SpinBox.setFont(font)
        self.ymax_SpinBox.setMinimum(-10.000000000000000)
        self.ymax_SpinBox.setMaximum(10.000000000000000)
        self.ymax_SpinBox.setValue(10.000000000000000)

        self.horizontalLayout_8.addWidget(self.ymax_SpinBox)

        self.ymax_hScrollBar = QScrollBar(self.layoutWidget)
        self.ymax_hScrollBar.setObjectName(u"ymax_hScrollBar")
        self.ymax_hScrollBar.setMinimumSize(QSize(150, 0))
        self.ymax_hScrollBar.setFont(font)
        self.ymax_hScrollBar.setMinimum(-10)
        self.ymax_hScrollBar.setMaximum(10)
        self.ymax_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_8.addWidget(self.ymax_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_10 = QLabel(self.layoutWidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMaximumSize(QSize(70, 16777215))
        self.label_10.setFont(font)
        self.label_10.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_9.addWidget(self.label_10)

        self.zmin_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.zmin_SpinBox.setObjectName(u"zmin_SpinBox")
        self.zmin_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.zmin_SpinBox.setFont(font)
        self.zmin_SpinBox.setMinimum(-10.000000000000000)
        self.zmin_SpinBox.setMaximum(10.000000000000000)
        self.zmin_SpinBox.setValue(-10.000000000000000)

        self.horizontalLayout_9.addWidget(self.zmin_SpinBox)

        self.zmin_hScrollBar = QScrollBar(self.layoutWidget)
        self.zmin_hScrollBar.setObjectName(u"zmin_hScrollBar")
        self.zmin_hScrollBar.setMinimumSize(QSize(150, 0))
        self.zmin_hScrollBar.setFont(font)
        self.zmin_hScrollBar.setMinimum(-10)
        self.zmin_hScrollBar.setMaximum(10)
        self.zmin_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_9.addWidget(self.zmin_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_11 = QLabel(self.layoutWidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setMaximumSize(QSize(70, 16777215))
        self.label_11.setFont(font)
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_10.addWidget(self.label_11)

        self.zmax_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.zmax_SpinBox.setObjectName(u"zmax_SpinBox")
        self.zmax_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.zmax_SpinBox.setFont(font)
        self.zmax_SpinBox.setMinimum(-10.000000000000000)
        self.zmax_SpinBox.setMaximum(10.000000000000000)
        self.zmax_SpinBox.setValue(10.000000000000000)

        self.horizontalLayout_10.addWidget(self.zmax_SpinBox)

        self.zmax_hScrollBar = QScrollBar(self.layoutWidget)
        self.zmax_hScrollBar.setObjectName(u"zmax_hScrollBar")
        self.zmax_hScrollBar.setMinimumSize(QSize(150, 0))
        self.zmax_hScrollBar.setFont(font)
        self.zmax_hScrollBar.setMinimum(-10)
        self.zmax_hScrollBar.setMaximum(10)
        self.zmax_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_10.addWidget(self.zmax_hScrollBar)


        self.verticalLayout_2.addLayout(self.horizontalLayout_10)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)


        self.horizontalLayout_14.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_12 = QLabel(self.layoutWidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setMaximumSize(QSize(100, 16777215))
        self.label_12.setFont(font)
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_11.addWidget(self.label_12)

        self.xslice_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.xslice_SpinBox.setObjectName(u"xslice_SpinBox")
        self.xslice_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.xslice_SpinBox.setFont(font)
        self.xslice_SpinBox.setMinimum(-10.000000000000000)
        self.xslice_SpinBox.setMaximum(10.000000000000000)
        self.xslice_SpinBox.setValue(0.000000000000000)

        self.horizontalLayout_11.addWidget(self.xslice_SpinBox)

        self.xslice_hScrollBar = QScrollBar(self.layoutWidget)
        self.xslice_hScrollBar.setObjectName(u"xslice_hScrollBar")
        self.xslice_hScrollBar.setMinimumSize(QSize(150, 0))
        self.xslice_hScrollBar.setFont(font)
        self.xslice_hScrollBar.setMinimum(-10)
        self.xslice_hScrollBar.setMaximum(10)
        self.xslice_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_11.addWidget(self.xslice_hScrollBar)


        self.verticalLayout_3.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_17 = QLabel(self.layoutWidget)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setMaximumSize(QSize(100, 16777215))
        self.label_17.setFont(font)
        self.label_17.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_12.addWidget(self.label_17)

        self.yslice_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.yslice_SpinBox.setObjectName(u"yslice_SpinBox")
        self.yslice_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.yslice_SpinBox.setFont(font)
        self.yslice_SpinBox.setMinimum(-10.000000000000000)
        self.yslice_SpinBox.setMaximum(10.000000000000000)
        self.yslice_SpinBox.setValue(0.000000000000000)

        self.horizontalLayout_12.addWidget(self.yslice_SpinBox)

        self.yslice_hScrollBar = QScrollBar(self.layoutWidget)
        self.yslice_hScrollBar.setObjectName(u"yslice_hScrollBar")
        self.yslice_hScrollBar.setMinimumSize(QSize(150, 0))
        self.yslice_hScrollBar.setFont(font)
        self.yslice_hScrollBar.setMinimum(-10)
        self.yslice_hScrollBar.setMaximum(10)
        self.yslice_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_12.addWidget(self.yslice_hScrollBar)


        self.verticalLayout_3.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_13 = QLabel(self.layoutWidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setMaximumSize(QSize(100, 16777215))
        self.label_13.setFont(font)
        self.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_13.addWidget(self.label_13)

        self.zslice_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.zslice_SpinBox.setObjectName(u"zslice_SpinBox")
        self.zslice_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.zslice_SpinBox.setFont(font)
        self.zslice_SpinBox.setMinimum(-10.000000000000000)
        self.zslice_SpinBox.setMaximum(10.000000000000000)
        self.zslice_SpinBox.setValue(0.000000000000000)

        self.horizontalLayout_13.addWidget(self.zslice_SpinBox)

        self.zslice_hScrollBar = QScrollBar(self.layoutWidget)
        self.zslice_hScrollBar.setObjectName(u"zslice_hScrollBar")
        self.zslice_hScrollBar.setMinimumSize(QSize(150, 0))
        self.zslice_hScrollBar.setFont(font)
        self.zslice_hScrollBar.setMinimum(-10)
        self.zslice_hScrollBar.setMaximum(10)
        self.zslice_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_13.addWidget(self.zslice_hScrollBar)


        self.verticalLayout_3.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.plane_comboBox = QComboBox(self.layoutWidget)
        self.plane_comboBox.addItem("")
        self.plane_comboBox.addItem("")
        self.plane_comboBox.addItem("")
        self.plane_comboBox.setObjectName(u"plane_comboBox")
        self.plane_comboBox.setMaximumSize(QSize(100, 16777215))
        font2 = QFont()
        font2.setFamilies([u"Helvetica"])
        font2.setPointSize(12)
        self.plane_comboBox.setFont(font2)

        self.horizontalLayout_18.addWidget(self.plane_comboBox)

        self.plane_SpinBox = QDoubleSpinBox(self.layoutWidget)
        self.plane_SpinBox.setObjectName(u"plane_SpinBox")
        self.plane_SpinBox.setMinimumSize(QSize(100, 0))
        self.plane_SpinBox.setMaximumSize(QSize(100, 16777215))
        self.plane_SpinBox.setFont(font)
        self.plane_SpinBox.setMinimum(-50.000000000000000)
        self.plane_SpinBox.setMaximum(50.000000000000000)

        self.horizontalLayout_18.addWidget(self.plane_SpinBox)

        self.plane_hScrollBar = QScrollBar(self.layoutWidget)
        self.plane_hScrollBar.setObjectName(u"plane_hScrollBar")
        self.plane_hScrollBar.setMinimumSize(QSize(0, 0))
        self.plane_hScrollBar.setFont(font)
        self.plane_hScrollBar.setMinimum(-10)
        self.plane_hScrollBar.setMaximum(10)
        self.plane_hScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_18.addWidget(self.plane_hScrollBar)


        self.verticalLayout_3.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_18 = QLabel(self.layoutWidget)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setMinimumSize(QSize(0, 28))
        self.label_18.setMaximumSize(QSize(120, 16777215))
        self.label_18.setFont(font)

        self.horizontalLayout_21.addWidget(self.label_18)

        self.tracer_index_SpinBox = QSpinBox(self.layoutWidget)
        self.tracer_index_SpinBox.setObjectName(u"tracer_index_SpinBox")
        self.tracer_index_SpinBox.setMinimumSize(QSize(120, 0))
        self.tracer_index_SpinBox.setMaximumSize(QSize(150, 16777215))
        self.tracer_index_SpinBox.setFont(font)

        self.horizontalLayout_21.addWidget(self.tracer_index_SpinBox)

        self.horizontalSpacer = QSpacerItem(320, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer)


        self.verticalLayout_3.addLayout(self.horizontalLayout_21)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_3.addItem(self.verticalSpacer)


        self.horizontalLayout_14.addLayout(self.verticalLayout_3)


        self.verticalLayout_7.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.savejpg_checkBox = QCheckBox(self.layoutWidget)
        self.savejpg_checkBox.setObjectName(u"savejpg_checkBox")
        self.savejpg_checkBox.setFont(font)
        self.savejpg_checkBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontalLayout_16.addWidget(self.savejpg_checkBox)

        self.start_animateButton = QPushButton(self.layoutWidget)
        self.start_animateButton.setObjectName(u"start_animateButton")
        self.start_animateButton.setMinimumSize(QSize(130, 30))
        font3 = QFont()
        font3.setFamilies([u"Helvetica"])
        font3.setPointSize(12)
        font3.setBold(True)
        self.start_animateButton.setFont(font3)

        self.horizontalLayout_16.addWidget(self.start_animateButton)

        self.stop_animateButton = QPushButton(self.layoutWidget)
        self.stop_animateButton.setObjectName(u"stop_animateButton")
        self.stop_animateButton.setMinimumSize(QSize(130, 30))
        self.stop_animateButton.setFont(font3)

        self.horizontalLayout_16.addWidget(self.stop_animateButton)

        self.continue_animateButton = QPushButton(self.layoutWidget)
        self.continue_animateButton.setObjectName(u"continue_animateButton")
        self.continue_animateButton.setMinimumSize(QSize(175, 30))
        self.continue_animateButton.setFont(font3)

        self.horizontalLayout_16.addWidget(self.continue_animateButton)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.cmap_checkBox = QCheckBox(self.layoutWidget)
        self.cmap_checkBox.setObjectName(u"cmap_checkBox")
        self.cmap_checkBox.setFont(font)
        self.cmap_checkBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontalLayout_20.addWidget(self.cmap_checkBox)

        self.label_16 = QLabel(self.layoutWidget)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setEnabled(True)
        self.label_16.setMaximumSize(QSize(50, 16777215))
        self.label_16.setFont(font)
        self.label_16.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_20.addWidget(self.label_16)

        self.cmap_LineEdit = QLineEdit(self.layoutWidget)
        self.cmap_LineEdit.setObjectName(u"cmap_LineEdit")
        self.cmap_LineEdit.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_20.addWidget(self.cmap_LineEdit)


        self.horizontalLayout_15.addLayout(self.horizontalLayout_20)

        self.cbar_checkBox = QCheckBox(self.layoutWidget)
        self.cbar_checkBox.setObjectName(u"cbar_checkBox")
        self.cbar_checkBox.setMaximumSize(QSize(170, 16777215))
        self.cbar_checkBox.setFont(font)
        self.cbar_checkBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontalLayout_15.addWidget(self.cbar_checkBox)

        self.label_4 = QLabel(self.layoutWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setEnabled(True)
        self.label_4.setMaximumSize(QSize(50, 16777215))
        self.label_4.setFont(font)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_15.addWidget(self.label_4)

        self.cbar_min_LineEdit = QLineEdit(self.layoutWidget)
        self.cbar_min_LineEdit.setObjectName(u"cbar_min_LineEdit")
        self.cbar_min_LineEdit.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_15.addWidget(self.cbar_min_LineEdit)

        self.label_15 = QLabel(self.layoutWidget)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setEnabled(True)
        self.label_15.setMaximumSize(QSize(50, 16777215))
        self.label_15.setFont(font)
        self.label_15.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_15.addWidget(self.label_15)

        self.cbar_max_LineEdit = QLineEdit(self.layoutWidget)
        self.cbar_max_LineEdit.setObjectName(u"cbar_max_LineEdit")
        self.cbar_max_LineEdit.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_15.addWidget(self.cbar_max_LineEdit)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_15)


        self.horizontalLayout_19.addLayout(self.horizontalLayout_16)


        self.verticalLayout_7.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_14 = QLabel(self.layoutWidget)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMinimumSize(QSize(0, 28))
        self.label_14.setMaximumSize(QSize(100, 16777215))
        self.label_14.setFont(font)

        self.horizontalLayout_17.addWidget(self.label_14)

        self.tframe_SpinBox = QSpinBox(self.layoutWidget)
        self.tframe_SpinBox.setObjectName(u"tframe_SpinBox")
        self.tframe_SpinBox.setMinimumSize(QSize(100, 0))
        self.tframe_SpinBox.setMaximumSize(QSize(200, 16777215))
        self.tframe_SpinBox.setFont(font)

        self.horizontalLayout_17.addWidget(self.tframe_SpinBox)

        self.tframe_hSlider = QSlider(self.layoutWidget)
        self.tframe_hSlider.setObjectName(u"tframe_hSlider")
        self.tframe_hSlider.setMinimumSize(QSize(0, 20))
        self.tframe_hSlider.setFont(font)
        self.tframe_hSlider.setOrientation(Qt.Orientation.Horizontal)
        self.tframe_hSlider.setTickPosition(QSlider.TickPosition.TicksBothSides)

        self.horizontalLayout_17.addWidget(self.tframe_hSlider)


        self.verticalLayout_7.addLayout(self.horizontalLayout_17)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)

        self.verticalSpacer_3 = QSpacerItem(1325, 13, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.verticalLayout_7.addItem(self.verticalSpacer_3)


        self.horizontalLayout_32.addLayout(self.verticalLayout_7)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1600, 37))
        self.menuFile = QMenu(self.menuBar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionClose_2)
        self.menuFile.addAction(self.actionQuit)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionClose_2.setText(QCoreApplication.translate("MainWindow", u"Close", None))
#if QT_CONFIG(shortcut)
        self.actionClose_2.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+W", None))
#endif // QT_CONFIG(shortcut)
        self.actionCtrl_W.setText(QCoreApplication.translate("MainWindow", u"Ctrl+W", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
#if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Q", None))
#endif // QT_CONFIG(shortcut)
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"File Type", None))
        self.filetype_comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"HDF5", None))
        self.filetype_comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"gda", None))

        self.plottype_comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Contour", None))
        self.plottype_comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Contour+X-Average", None))
        self.plottype_comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Contour+X-Slice", None))
        self.plottype_comboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"Contour+Z-Average", None))
        self.plottype_comboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"Contour+Z-Slice", None))

        self.integrate_checkBox.setText(QCoreApplication.translate("MainWindow", u"Integrate along Y", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Raw Plot", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Diagnostic Plot", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Overplot", None))
        self.autoplot_checkBox.setText(QCoreApplication.translate("MainWindow", u"Auto Update Plot", None))
        self.plotButton.setText(QCoreApplication.translate("MainWindow", u"Plot", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"X-Min", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"X-Max", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Y-Min", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Y-Max", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Z-Min", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Z-Max", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"X-Slice", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Y-Slice", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Z-Slice", None))
        self.plane_comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"X-Plane", None))
        self.plane_comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Y-Plane", None))
        self.plane_comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Z-Plane", None))

        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Tracer Index", None))
        self.savejpg_checkBox.setText(QCoreApplication.translate("MainWindow", u"SAVE JPEGs", None))
        self.start_animateButton.setText(QCoreApplication.translate("MainWindow", u"Start Animation", None))
        self.stop_animateButton.setText(QCoreApplication.translate("MainWindow", u"Stop Animation", None))
        self.continue_animateButton.setText(QCoreApplication.translate("MainWindow", u"Continue Animation", None))
        self.cmap_checkBox.setText(QCoreApplication.translate("MainWindow", u"Fix Colormap", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"cmap: ", None))
        self.cbar_checkBox.setText(QCoreApplication.translate("MainWindow", u"Fix Colorbar Range", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Min: ", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Max: ", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Time Slice", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

