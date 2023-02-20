
from PyQt5.QtCore import QDir, Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QVBoxLayout, QDockWidget, QGridLayout, QWidget)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import PIL
from PIL import ImageQt

import cv2
import numpy as np
import os
import insightface
import urllib
import urllib.request
import copy
import mxnet as mx
import pickle

from FRPipeline import FRP

cap = cv2.VideoCapture(0)
num_subjects = 0

#face recognition pipeline parameters
use_gpu=True
detection_algorithm = FRP.DETECT_MOBILE_MNET
alignment_algorithm = FRP.ALIGN_INSIGHTFACE
recognition_algorithm = FRP.RECOGNITION_SHUFFLEFACENET

fr_pipeline = FRP(use_gpu, detection_algorithm, alignment_algorithm, recognition_algorithm,db_filename="new_test_database.pickle")



class CameraViewer(QMainWindow):
    def __init__(self):
        super(CameraViewer, self).__init__()
        
        
        self.saving_capture = False
        self.counter = 0
        
        #interface handling
        masterWidget = QWidget()
        layout = QGridLayout()
        
        
        l1 = QLabel()

        l1.setText("List of identities \n" + '\n'.join(list(fr_pipeline.labels[:,1])))
        l1.setAlignment(Qt.AlignCenter)
        self.statusLabel = QLabel()
        self.statusLabel.setText("Waiting for faces...")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        
        
       
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setFixedSize(640,480)
        
        
        layout.addWidget(self.imageLabel,0,0)
        layout.addWidget(l1,0,1)
        layout.addWidget(self.statusLabel,1,0)
        
        masterWidget.setLayout(layout)
       
        self.setCentralWidget(masterWidget)
        
        self.setWindowTitle("Live Camera Viewer")
        self.createActions()
        self.createMenus()

        
        #self.resize(1280, 800)
    
        timer = QTimer(self)
        timer.timeout.connect(self.open)
        timer.start(33) #30 Hz

    def toggleSave(self):
        if self.saving_capture:
            self.saving_capture=False
            self.counter = 0
        else:
            self.saving_capture=True
            self.enrollDatabase()
            
    def enrollDatabase(self):
        num_subjects+=1
        dir_name = str(num_subjects)
        os.mkdir(os.path.join("images",dir_name))
        
        #cv2.imwrite(os.path.join(dir_name, face_file_name), image)
        

    def createActions(self):
        #self.saveAct = QAction("&Toggle Save...", self, shortcut="Ctrl+O",
        #        triggered=self.toggleSave)
        self.saveEnroll = QAction("&Enroll subject on database...", self, shortcut="Ctrl+1",
                triggered=self.enrollDatabase)
        
    def createMenus(self):
        self.demoMenu = QMenu("&Demo", self)
        
        
        #self.demoMenu.addAction(self.saveAct)
        self.demoMenu.addAction(self.saveEnroll)
        
        self.menuBar().addMenu(self.demoMenu)
        
    #MAIN FRAME LOOP    
    def open(self):
        #get data and display
        ret, frame = cap.read()
        
        im_bgr = frame #taken from opencv camera

        #full pipeline
        bbox, lmk, ids, scores, embeddings = fr_pipeline.executeFullPipeline(im_bgr, format="BGR")

        #face detection step


        #crop
        resized_landmarks = copy.deepcopy(lmk)

        detected_faces = bbox.shape[0]

        if detected_faces:
            self.statusLabel.setText("Detecting " + str(detected_faces) + (" face!" if (detected_faces == 1) else " faces!"))
        else:
            self.statusLabel.setText("Waiting for faces...")

        #iterate
        for face_idx in range(detected_faces):
            corner1 = (bbox[face_idx][0],bbox[face_idx][1]) #(x,y)
            corner2 = (bbox[face_idx][2],bbox[face_idx][3]) #(x,y)
            
            cv2.rectangle(im_bgr,corner1,corner2,(255,0,0),2)
            
            #warp landmarks to resized face region
            #112 or 224 used for insightface util alignment
            #ratio_width = FRP.ALIGNMENT_RESOLUTION/float(np.shape(face)[1])
            #ratio_height = FRP.ALIGNMENT_RESOLUTION/float(np.shape(face)[0])
            
            #cropped_landmarks = copy.deepcopy(lmk[face_idx,:,:])
            #cropped_landmarks[:,0] = cropped_landmarks[:,0] - min(bbox[face_idx][0],bbox[face_idx][2])
            #cropped_landmarks[:,1] = cropped_landmarks[:,1] - min(bbox[face_idx][1],bbox[face_idx][3])
                
            #resized_landmarks[face_idx,:,0] = (cropped_landmarks[:,0]*ratio_width).astype(np.int)
            #resized_landmarks[face_idx,:,1] = (cropped_landmarks[:,1]*ratio_height).astype(np.int)
                
            #draw landmarks
            #for lmk in range(5):
            #    face = cv2.circle(face,(cropped_landmarks[lmk,0],cropped_landmarks[lmk,1]),10,(0,255,0),2)
            #    resized_face = cv2.circle(resized_face,(resized_landmarks[face_idx,lmk,0],resized_landmarks[face_idx,lmk,1]),10,(0,255,0),2)
                    
            #face alignment step
            #aligned_face = fr_pipeline.alignCropFace(resized_face, resized_landmarks[face_idx, :, :])

            #face recognition step
            #sim, identity = fr_pipeline.recognizeSingleFace(aligned_face)
            sim = scores[face_idx]
            identity = ids[face_idx]

            if identity != -1: #TODO: Adjust recognition threshold
                cv2.putText(im_bgr, fr_pipeline.labels[identity][1]+ ' ' + str(round(sim,3)), (bbox[face_idx,0],bbox[face_idx,1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                #print(sim)
            if (self.saving_capture):
                #cv2.imwrite(os.path.join('images',str(num_subjects),str(self.counter)+'.jpg'),face)
                #cv2.imwrite(os.path.join('images',str(num_subjects),'resized_'+str(self.counter)+'.jpg'),resized_face)
                #cv2.imwrite(os.path.join('images',str(num_subjects),'aligned_'+str(self.counter)+'.jpg'),aligned_face)
                self.counter+=1
                if (self.counter==30):
                    self.toggleSave()
        
        #convert image from bgr to rgb for GUI handling
        im_rgb = im_bgr[:, :, [2, 1, 0]]    
        pilimg = PIL.Image.fromarray(im_rgb,"RGB")
        image = ImageQt.ImageQt(pilimg)
            
        if image.isNull():
            QMessageBox.information(self, "Live Camera Viewer","Cannot load camera")
            return
        
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.imageLabel.adjustSize()



class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(500, 400)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                "<p>The <b>Image Viewer</b> example shows how to combine "
                "QLabel and QScrollArea to display an image. QLabel is "
                "typically used for displaying text, but it can also display "
                "an image. QScrollArea provides a scrolling view around "
                "another widget. If the child widget exceeds the size of the "
                "frame, QScrollArea automatically provides scroll bars.</p>"
                "<p>The example demonstrates how QLabel's ability to scale "
                "its contents (QLabel.scaledContents), and QScrollArea's "
                "ability to automatically resize its contents "
                "(QScrollArea.widgetResizable), can be used to implement "
                "zooming and scaling features.</p>"
                "<p>In addition the example shows how to use QPainter to "
                "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    CameraViewer = CameraViewer()
    CameraViewer.show()
    #imageViewer = ImageViewer()
    #imageViewer.show()
    sys.exit(app.exec_())