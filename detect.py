# -*- coding: utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model='yolo11n.pt')
    model.predict(source='mydata/test/images',
                  save=True,
                  show=True,
                  )
