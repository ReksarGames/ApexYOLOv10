import win32gui
import win32con
import win32api


def show_target(box):
    hwnd = win32gui.GetDesktopWindow()
    hwndDC = win32gui.GetDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文
    pen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 255))  # 定义框颜色
    brush = win32gui.GetStockObject(win32con.NULL_BRUSH)  # 定义透明画刷，这个很重要！！

    win32gui.SelectObject(hwndDC, pen)
    win32gui.SelectObject(hwndDC, brush)
    win32gui.Rectangle(hwndDC, box[0], box[1], box[2], box[3])  # 左上到右下的坐标

    win32gui.ReleaseDC(hwnd, hwndDC)
    return
