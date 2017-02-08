import numpy as np
import cv2

class ImageStackChart(object):
    def __init__(self, scale=1., title='image stack', fps=60.):
        self.im = None
        self.scale = scale
        self.fps = fps
        self.title = title

    def update(self, conv_output):
        im = np.transpose(conv_output, [2, 0, 1])
        # normalize per image
        min_im = im.min(1, keepdims=True).min(2, keepdims=True)
        ptp_im = im.ptp(1).ptp(1).reshape((-1, 1, 1))
        im = (im - min_im) / ptp_im
        # stack images horizonally and scale
        self.im = cv2.resize(np.hstack(im), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

    def draw(self):
        render_image(self.title, np.expand_dims(self.im, -1) * 255., self.fps, True)

# common utilities
def render_image(title, im, render_fps, is_grayscale, input_color_format='RGB'):
    img = np.asarray(im, dtype='uint8')
    if is_grayscale:
        img = np.squeeze(img, -1)
    elif input_color_format == 'RGB':
        # cv2.imshow assumes BGR format
        # this flips the bytes in RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img)
    cv2.waitKey(int(1000. / render_fps))
