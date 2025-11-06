from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO(model='yolo11n-GAM.yaml')
    model = YOLO(model='yolo11n.yaml')

    # results = model.train(data="coco128.yaml", epochs=20, batch=16, device=0)
    results = model.train(data="coco8.yaml", epochs=20, batch=16, device=0)