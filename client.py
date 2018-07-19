import socket
import time
from imagezmq import imagezmq
import cv2
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', help='port to use', default='5555',type=str)
    parser.add_argument('--host', help='server IP or name', default='localhost', required=False)
    args = parser.parse_args()

    sender = imagezmq.ImageSender(connect_to='tcp://'+args.host+':' + args.port)

    cap = cv2.VideoCapture(0)

    while True:  # send images as stream until Ctrl-C

        ret, frame = cap.read()

        if not ret:
            print("Unable to open camera")
            return


        image = cv2.flip(frame, 1)

        mst, img = sender.send_image2("image", image)


        c = img.shape[1] * 1
        r = img.shape[0] * 1

        img = cv2.resize(img, dsize=(c, r), interpolation=cv2.INTER_CUBIC)


        cv2.imshow('test',img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
