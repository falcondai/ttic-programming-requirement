# visualizing time series, such as value estimate and action entropy, with
# horizon charts.

# basically we fold the time series into a fixed height area chart.
# using the ideas from https://square.github.io/cubism/

import numpy as np
from image import render_image

def fold(series, resolution, height, baseline=0.):
    '''
    resolution: value per pixel, this is used to convert [float] into [int]
    height: height of the plot in pixels
    baseline: where y = 0 in the plot corresponds to in value space
    '''
    yi = series - baseline
    si = np.floor_divide(yi, resolution * height)
    ypi = np.floor_divide(yi - si * resolution * height, resolution)
    return np.asarray(si, dtype=np.int32), np.asarray(ypi, dtype=np.int32)

def fold_y(y, resolution, height, baseline=0.):
    '''
    resolution: value per pixel, this is used to convert [float] into [int]
    height: height of the plot in pixels
    baseline: where y = 0 in the plot corresponds to in value space
    '''
    yi = y - baseline
    si = np.floor_divide(yi, resolution * height)
    ypi = np.floor_divide(yi - si * resolution * height, resolution)
    return int(si), int(ypi)

def blend(color1, color2, alpha):
    return np.asarray(color1 * alpha + color2 * (1. - alpha), dtype=np.uint8)

def blend_ratio(x):
    return (1. - np.exp(- 0.4 * x))

def shade(step, positive_color=np.asarray([255, 0, 0]), negative_color=np.asarray([0, 0, 255]), background_color=np.asarray([255, 255, 255])):
    if step >= 0:
        color = blend(positive_color, background_color, blend_ratio(step + 1.))
    else:
        color = blend(negative_color, background_color, blend_ratio(-step))
    return color

class HorizonChart(object):
    def __init__(self, length, resolution, height, baseline=0., positive_color=[0, 0, 255], negative_color=[255, 0, 0], background_color=[255, 255, 255], title='horizon', fps=60.):
        self.length = int(length)
        self.height = int(height)
        self.resolution = resolution
        self.baseline = baseline
        self.positive_color = np.asarray(positive_color, dtype=np.uint8)
        self.negative_color = np.asarray(negative_color, dtype=np.uint8)
        self.background_color = np.asarray(background_color, dtype=np.uint8)

        self.im = np.zeros((height, length, 3), dtype=np.uint8)
        self.im[:, :] = self.background_color

        self.fps = fps
        self.title = title

    def update(self, y):
        s, yp = fold_y(y, self.resolution, self.height, self.baseline)
        c = shade(s, self.positive_color, self.negative_color, self.background_color)
        # shift chart to the left by 1 pixel
        self.im[:, :-1] = self.im[:, 1:]
        self.im[:, -1] = self.background_color
        if s >= 0:
            self.im[self.height-yp:, -1] = c
            if s > 0:
                self.im[:self.height-yp, -1] = shade(s - 1, self.positive_color, self.negative_color, self.background_color)
        else:
            self.im[:self.height-yp, -1] = c
            if s < -1:
                self.im[self.height-yp:, -1] = shade(s + 1, self.positive_color, self.negative_color, self.background_color)

    def draw(self):
        render_image(self.title, self.im, self.fps, False)


if __name__ == '__main__':
    import cv2

    # foo = np.arange(1000) / 100.
    foo = [0.]
    for _ in xrange(1000):
        foo.append(foo[-1] + np.random.rand() - 0.5)
    foo = np.asarray(foo)
    n = len(foo)
    h = 100
    # si, yi = fold(foo, 0.06, h, 0)
    si, yi = [], []
    for y in foo:
        s, yp = fold_y(y, 0.04, h, 0)
        si.append(s)
        yi.append(yp)
        # print s, yp

    im = np.ones((h, n, 3), dtype=np.uint8) * 255
    for x in xrange(n):
        c = shade(si[x])
        if si[x] >= 0:
            im[h-yi[x]:, x] = c
            if si[x] > 0:
                im[:h-yi[x], x] = shade(si[x] - 1)
        else:
            im[:h-yi[x], x] = c
            if si[x] < -1:
                im[h-yi[x]:, x] = shade(si[x] + 1)

        print shade(si[x])

    cv2.imshow('plot', im)
    cv2.waitKey(0)
