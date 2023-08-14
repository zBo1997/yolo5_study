import torch, pynput
import numpy as np
import win32gui, win32con, cv2
from grabscreen import grab_screen_win32 # 本地文件
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh


# 可调参数
conf_thres = 0.25
iou_thres = 0.05
thickness = 2
x, y = (1920, 1080)
re_x, re_y = (1920, 1080)



def LoadModule():
    device = select_device('')
    weights = 'best.pt'
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    return model



if __name__ == '__main__':
    model = LoadModule()
    while True:
        names = model.names
        img0 = grab_screen_win32(region=(0, 0, 1920, 1080))

        im = letterbox(img0, 640, stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False,
                                   max_det=1000)
        boxs=[]
        for i, det in enumerate(pred):  # per image
            im0 = img0.copy()
            s = ' '
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = img0  # for save_crop

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                box = ('%g ' * len(line)).rstrip() % line
                box = box.split(' ')
                boxs.append(box)
            if len(boxs):
                for i, det in enumerate(boxs):
                    _, x_center, y_center, width, height = det
                    x_center, width = re_x * float(x_center), re_x * float(width)
                    y_center, height = re_y * float(y_center), re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                    color = (0, 0, 255)  # RGB
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=thickness)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow()
            break
        cv2.namedWindow('windows', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('windows', re_x // 2, re_y // 2)
        cv2.imshow('windows', img0)
        HWND = win32gui.FindWindow(None, "windows")
        win32gui.SetWindowPos(HWND, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)


