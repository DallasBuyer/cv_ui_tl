# -*- coding: utf-8 -*-

import tensorflow as tf
import model_utils
import sys
from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtGui import QWidget
import numpy as np
import preprocess_imagess
from scipy.misc import imread, imresize

class MainWidget(QWidget):
    def __init__(self):
        super(MainWidget, self).__init__()

        reload(sys)
        sys.setdefaultencoding("utf-8")

        self.setWindowTitle("Image Classification")
        palette = QtGui.QPalette()
        palette.setColor(self.backgroundRole(), Qt.QColor(233, 233, 233))
        # palette.setColor(self.foregroundRole(), Qt.QColor(255, 255, 255))
        self.setPalette(palette)
        self.init_ui()

        # self.graph = tf.Graph()


    def init_ui(self):

        # self.palette_color = QtGui.QPalette()
        # self.palette_color.setColor(self.foregroundRole(), Qt.QColor(255, 255, 255))

        self.createGridGroupBox_1()
        self.createGridGroupBox_2()
        self.createGridGroupBox_3()
        self.createGridGroupBox_4()
        self.creatFormGroupBox()
        self.createGridGroupBox_5()
        self.createGridGroupBox_6()

        mainLayout = QtGui.QVBoxLayout()
        hboxLayout_1 = QtGui.QHBoxLayout()
        hboxLayout_2 = QtGui.QHBoxLayout()
        hboxLayout_3 = QtGui.QHBoxLayout()
        hboxLayout_4 = QtGui.QHBoxLayout()

        #hboxLayout_1.addStretch()
        hboxLayout_1.addWidget(self.gridGroupBox_1)
        hboxLayout_1.addWidget(self.gridGroupBox_2)

        hboxLayout_2.addWidget(self.gridGroupBox_3)
        hboxLayout_3.addWidget(self.gridGroupBox_4)
        hboxLayout_4.addWidget(self.gridGroupBox_5)
        hboxLayout_4.addWidget(self.gridGroupBox_6)

        mainLayout.addLayout(hboxLayout_1)
        mainLayout.addLayout(hboxLayout_2)
        mainLayout.addLayout(hboxLayout_3)
        # mainLayout.addWidget(self.formGroupBox)
        mainLayout.addLayout(hboxLayout_4)
        self.setLayout(mainLayout)

    def createGridGroupBox_1(self):
        self.gridGroupBox_1 = QtGui.QGroupBox(u"数据及模型选择")
        layout = QtGui.QGridLayout()

        self.data_combox = QtGui.QComboBox()
        self.data_combox.addItem("mnist_28_28_1")
        self.data_combox.addItem("mnist_32_32_1")
        self.data_combox.addItem("mnist_224_224_1")
        self.data_combox.addItem("cifar10_32_32_3")
        self.data_combox.addItem("ImageNet")
        self.data_label = QtGui.QLabel("datasets: ")
        self.data_label.setBuddy(self.data_combox)
        self.data_current_label = QtGui.QLabel("current dataset:")
        self.data_current_text = QtGui.QLineEdit("MNIST")
        self.data_current_text.setEnabled(False)
        self.data_button = QtGui.QPushButton("load data")

        self.model_combox = QtGui.QComboBox()
        self.model_combox.addItem("MLP")
        self.model_combox.addItem("CNN_28_28_1")
        self.model_combox.addItem("CNN_32_32_3")
        self.model_combox.addItem("CNN_32_32_1")
        self.model_combox.addItem("AlexNet_28_28_1")
        self.model_combox.addItem("AlexNet_32_32_3")
        self.model_combox.addItem("VGG16_224_224_3")
        self.model_combox.addItem("Inception_v1_224_224_1")
        self.model_combox.addItem("Inception_v1_224_224_3")
        self.model_combox.addItem("ResNet")
        self.model_label = QtGui.QLabel("model: ")
        self.model_label.setBuddy(self.model_combox)
        self.model_current_label = QtGui.QLabel("current model:")
        self.model_current_text = QtGui.QLineEdit("MLP")
        self.model_current_text.setEnabled(False)
        # set color
        # self.model_current_text.setAutoFillBackground(True)
        # model_palette = QtGui.QPalette()
        # model_palette.setColor(self.model_current_text.backgroundRole(), QtGui.QColor(34, 35,222))
        # self.model_current_text.setPalette(model_palette)
        self.model_button = QtGui.QPushButton("select model")

        layout.addWidget(self.data_label, 0, 0)
        layout.addWidget(self.data_combox, 0, 1, 1, 3)
        layout.addWidget(self.model_label, 1, 0)
        layout.addWidget(self.model_combox, 1, 1, 1, 3)
        layout.addWidget(self.data_current_label, 2, 0)
        layout.addWidget(self.data_current_text, 2, 1, 1, 3)
        layout.addWidget(self.model_current_label, 3, 0)
        layout.addWidget(self.model_current_text, 3, 1, 1, 3)
        layout.addWidget(self.data_button, 4, 0, 1, 4)
        layout.addWidget(self.model_button, 5, 0, 1, 4)

        self.gridGroupBox_1.setLayout(layout)

        self.connect(self.data_button, QtCore.SIGNAL('clicked()'), self.load_data)
        self.connect(self.model_button, QtCore.SIGNAL('clicked()'), self.load_model)

    def load_data(self):
        # prepare the data for training
        self.current_data = self.data_combox.currentText()
        print "load dataset: ", self.current_data
        self.data_current_text.setText(self.current_data)

    def load_model(self):
        # get the network
        self.current_model = self.model_combox.currentText()
        print "load model: ", self.current_model
        self.model_current_text.setText(self.current_model)

    def createGridGroupBox_2(self):
        self.gridGroupBox_2 = QtGui.QGroupBox(u"训练选择参数")
        layout = QtGui.QGridLayout()
        n_epoch_label = QtGui.QLabel("epoch number:")
        self.n_epoch_text = QtGui.QLineEdit("10")
        batch_size_label = QtGui.QLabel("batch size:")
        self.batch_size_text = QtGui.QLineEdit("128")
        learn_rate_label = QtGui.QLabel("learning rate:")
        self.learn_rate_text = QtGui.QLineEdit("0.001")
        print_freq_label = QtGui.QLabel("print frequency/n_epoch")
        self.print_freq_text = QtGui.QLineEdit("2")
        save_n_label = QtGui.QLabel("save model/n_epoch:")
        self.save_n_text = QtGui.QLineEdit("5")
        self.reset_button = QtGui.QPushButton("reset")
        self.set_button = QtGui.QPushButton("set")

        layout.addWidget(n_epoch_label, 0, 0)
        layout.addWidget(self.n_epoch_text, 0, 1, 1, 2)
        layout.addWidget(batch_size_label, 1, 0)
        layout.addWidget(self.batch_size_text, 1, 1, 1, 2)
        layout.addWidget(learn_rate_label, 2, 0)
        layout.addWidget(self.learn_rate_text, 2, 1, 1, 2)
        layout.addWidget(print_freq_label, 3, 0)
        layout.addWidget(self.print_freq_text, 3, 1, 1, 2)
        layout.addWidget(save_n_label, 4, 0)
        layout.addWidget(self.save_n_text, 4, 1, 1, 2)
        layout.addWidget(self.reset_button, 5, 0)
        layout.addWidget(self.set_button, 5, 1, 1, 2)

        self.gridGroupBox_2.setLayout(layout)
        self.connect(self.set_button, QtCore.SIGNAL('clicked()'), self.set_params)
        self.connect(self.reset_button, QtCore.SIGNAL('clicked()'), self.reset_params)

    def set_params(self):
        self.n_epoch = int(self.n_epoch_text.text())
        self.batch_size = int(self.batch_size_text.text())
        self.learning_rate = float(self.learn_rate_text.text())
        self.print_freq = int(self.print_freq_text.text())
        self.save_n = int(self.save_n_text.text())  # save the model params per n_epoch
        print "n_epoch: ",self.n_epoch, "batch_size: ", self.batch_size, "learning_rate: ", self.learning_rate, \
            "print_freq: ", self.print_freq, "save_n: ", self.save_n

    def reset_params(self):
        self.n_epoch_text.setText("10")
        self.batch_size_text.setText("128")
        self.learn_rate_text.setText("0.001")
        self.print_freq_text.setText("2")
        self.save_n_text.setText("5")
        self.set_params()

    def creatFormGroupBox(self):
        self.formGroupBox = QtGui.QGroupBox(u"程序输出")
        layout = QtGui.QFormLayout()
        planEditor = QtGui.QTextEdit()
        planEditor.setPlainText(u"请选择上面训练或者测试操作，点击相应按钮")
        planEditor.setFixedHeight(150)
        layout.addRow(planEditor)
        self.formGroupBox.setLayout(layout)

    def createGridGroupBox_3(self):
        self.gridGroupBox_3 = QtGui.QGroupBox(u"优化函数选择")
        layout = QtGui.QGridLayout()

        self.opt_radio_1 = QtGui.QRadioButton("GradientDescent")
        self.opt_radio_2 = QtGui.QRadioButton("Momentum optimize")
        self.opt_radio_3 = QtGui.QRadioButton("Adagrad Optimize")
        self.opt_radio_4 = QtGui.QRadioButton("Adadelta Optimize")
        self.opt_radio_5 = QtGui.QRadioButton("RMSprop Optimize")
        self.opt_radio_6 = QtGui.QRadioButton("Adam Optimize")
        self.opt_radio_7 = QtGui.QRadioButton("Proximal Adagrad")
        self.opt_radio_8 = QtGui.QRadioButton("Proximal GradDescent")
        # self.opt_radio_1.setPalette(self.palette_color)
        self.opt_radio_6.setChecked(True)

        layout.addWidget(self.opt_radio_1, 0, 0)
        layout.addWidget(self.opt_radio_2, 0, 1)
        layout.addWidget(self.opt_radio_3, 0, 2)
        layout.addWidget(self.opt_radio_4, 0, 3)
        layout.addWidget(self.opt_radio_5, 1, 0)
        layout.addWidget(self.opt_radio_6, 1, 1)
        layout.addWidget(self.opt_radio_7, 1, 2)
        layout.addWidget(self.opt_radio_8, 1, 3)

        self.gridGroupBox_3.setLayout(layout)

    def opt_radio_click(self):
        clicked_list = [self.opt_radio_1.isChecked(), self.opt_radio_2.isChecked(), self.opt_radio_3.isChecked(),
                        self.opt_radio_4.isChecked(), self.opt_radio_5.isChecked(), self.opt_radio_6.isChecked(),
                        self.opt_radio_7.isChecked(), self.opt_radio_8.isChecked()]
        print clicked_list, clicked_list.index(1)
        return clicked_list.index(1)

    def createGridGroupBox_4(self):
        self.gridGroupBox_4 = QtGui.QGroupBox(u"操作选择")
        layout = QtGui.QGridLayout()

        self.train_button = QtGui.QPushButton("train model")
        self.pre_load_check = QtGui.QCheckBox("load pre_trained model")
        self.save_check = QtGui.QCheckBox("save trained model")
        self.test_button = QtGui.QPushButton("test model")
        # self.pre_load_check.setPalette(self.palette_color)

        layout.addWidget(self.train_button, 0, 0)
        layout.addWidget(self.pre_load_check, 0, 1)
        layout.addWidget(self.save_check, 0, 2)
        layout.addWidget(self.test_button, 0, 3)

        self.pre_load = True
        self.is_save = True
        self.pre_load_check.setChecked(True)
        self.save_check.setChecked(True)

        self.gridGroupBox_4.setLayout(layout)
        self.connect(self.train_button, QtCore.SIGNAL('clicked()'), self.question_train)
        self.connect(self.test_button, QtCore.SIGNAL('clicked()'), self.question_test)
        self.connect(self.pre_load_check, QtCore.SIGNAL('clicked()'), self.pre_load_model)
        self.connect(self.save_check, QtCore.SIGNAL('clicked()'), self.save_model)


    def pre_load_model(self):
        if self.pre_load_check.checkState():
            self.pre_load = True
        else:
            self.pre_load = False
        print "whether to load the pre_trained model: ", self.pre_load

    def save_model(self):
        if self.save_check.checkState():
            self.is_save = True
        else:
            self.is_save = False
        print "whether to save the model during training process: ", self.is_save

    def question_train(self):
        reply = QtGui.QMessageBox.question(self, "Confirm Training",
                "Are you confirmed to train the model",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            opt_index = self.opt_radio_click()
            self.train_button.setDisabled(True)
            # with self.graph.as_default():
            self.optimizer = model_utils.select_optimization(opt_index, self.learning_rate)
            print "select optimizer: ", self.optimizer.get_name()

            from train_thread import TrainThread
            self.train_thread = TrainThread(self.current_data, self.current_model,self.pre_load,
                                            self.is_save, self.batch_size, self.n_epoch, self.print_freq,
                                            self.save_n,self.optimizer, tf.get_default_graph())

            self.train_thread.finish_signal.connect(self.train_task_end)
            self.train_thread.start()
        else:
            pass

    def question_test(self):
        reply = QtGui.QMessageBox.question(self, "Confirm Test",
                "Are you confirmed to test the model",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.test_button.setDisabled(True)
            from test_thread import TestThread
            self.test_thread = TestThread(self.current_model, tf.get_default_graph())
            self.test_thread.finish_signal.connect(self.test_task_end)
            self.test_thread.start()
        else:
            pass

    def train_task_end(self, ls):
        print "*"*20
        from plot_thread import PlotThread
        self.plot_thread = PlotThread(ls)
        self.plot_thread.start()
        self.train_button.setDisabled(False)

    def test_task(self):
        pass

    def test_task_end(self):
        self.test_button.setDisabled(False)

    def createGridGroupBox_5(self):
        self.gridGroupBox_5 = QtGui.QGroupBox(u"测试模型")
        self.gridGroupBox_5.setFixedSize(230, 300)
        layout = QtGui.QVBoxLayout()
        layout_up = QtGui.QGridLayout()

        self.propTable = QtGui.QTableWidget(self)
        self.propTable.setColumnCount(2)
        self.propTable.setRowCount(5)
        self.propTable.setHorizontalHeaderLabels(['class', 'prop'])
        self.propTable.setItem(0, 0, QtGui.QTableWidgetItem(u"游轮"))
        self.propTable.setItem(1, 0, QtGui.QTableWidgetItem(u"货轮"))
        self.propTable.setItem(2, 0, QtGui.QTableWidgetItem(u"舰艇"))
        self.propTable.setItem(3, 0, QtGui.QTableWidgetItem(u"渔船"))
        self.propTable.setItem(4, 0, QtGui.QTableWidgetItem(u"快艇"))
        self.propTable.setFixedSize(200, 177)
        self.propTable.setColumnWidth(0, 90)
        self.propTable.setColumnWidth(1, 90)
        for i in [0, 1, 2, 3, 4]:
            self.propTable.setRowHeight(i, 30)

        self.open_button = QtGui.QPushButton("open")
        # self.open_button.setFixedWidth(200)

        self.test_label = QtGui.QLabel("dataset:")
        self.test_dataset = QtGui.QComboBox()
        self.test_dataset.addItem("MNIST")
        self.test_dataset.addItem("ImageNet")
        self.test_dataset.addItem("cifar10")
        layout_up.addWidget(self.test_label, 0, 0, 1, 1)
        layout_up.addWidget(self.test_dataset,0, 1, 1, 3)

        layout.addWidget(self.open_button)
        layout.addLayout(layout_up)
        layout.addWidget(self.propTable)

        self.gridGroupBox_5.setLayout(layout)
        self.connect(self.open_button, QtCore.SIGNAL('clicked()'), self.open)


    def createGridGroupBox_6(self):
        self.gridGroupBox_6 = QtGui.QGroupBox(u"测试图片")
        layout = QtGui.QHBoxLayout()

        self.picView = QtGui.QLabel(self)
        self.picView.setBackgroundRole(QtGui.QPalette.Dark)
        self.picView.setSizePolicy(QtGui.QSizePolicy.Ignored,
                                   QtGui.QSizePolicy.Ignored)
        self.picView.setScaledContents(True)

        image = QtGui.QImage("/home/xupeng/code/pycharm/tensorflow-models/imgrec-googlenet/doc/image_0423.jpg")
        print image.size()
        pixmap = QtGui.QPixmap.fromImage(image)
        print pixmap.size()
        self.picView.setPixmap(pixmap)
        self.picView.setFixedSize(400, 255)

        layout.addWidget(self.picView)

        self.gridGroupBox_6.setLayout(layout)

    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                "Open File", QtCore.QDir.currentPath())
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Recognition",
                    "Cannot load %s. " % fileName)
                return
        pixmap = QtGui.QPixmap.fromImage(image)
        self.picView.setPixmap(pixmap)
        print fileName

        size = model_utils.model_size(self.current_model)
        image = preprocess_imagess.image_preprocess(str(fileName), size)

        props = self.test_thread.predict(img=image)
        preds = (np.argsort(props)[::-1])[0:5]
        # for i in range(len(props)):
        #     props[i] = round(props[i])
        self.test_data_name = self.test_dataset.currentText()
        class_name = model_utils.load_name(self.test_data_name)
        for p in preds:
            print class_name[p], props[p]
        for i in range(5):
            self.propTable.setItem(i, 0, QtGui.QTableWidgetItem(str(class_name[preds[i]])))
            self.propTable.setItem(i, 1, QtGui.QTableWidgetItem(str(props[preds[i]])))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ships_recognition = MainWidget()
    ships_recognition.show()
    sys.exit(app.exec_())



