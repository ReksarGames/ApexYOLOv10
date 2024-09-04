from Apex.Grabber import Grabber
import cv2


def test_grabber():
    print("Инициализация Grabber...")
    grabber = Grabber()
    grabber.obs_vc_init(camera_name="OBS Virtual Camera")

    print("Начало захвата изображения...")
    while True:
        frame = grabber.get_image()
        if frame is None:
            print("Не удалось захватить изображение. Возможно, камера не инициализирована или неактивна.")
            break

        cv2.imshow("Камера OBS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_grabber()
