from matplotlib.widgets import RadioButtons

import matplotlib.pyplot as plt

class Visualiser:

    def select_x(self, label):
        self.feature_x = self.feature_names.index(label)
        plt.sca(self.plot_axes)
        self.show_cut()

    def select_y(self, label):
        self.feature_y = self.feature_names.index(label)
        plt.sca(self.plot_axes)
        self.show_cut()

    def __init__(self, data, min_rect, max_rect, feature_x, feature_y, feature_names):
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.min_rect = min_rect
        self.max_rect = max_rect
        self.data = data
        if isinstance(feature_names, list):
            self.feature_names = feature_names
        else:
            self.feature_names = feature_names.tolist()

        f, (self.plot_axes) = plt.subplots()
        f_menu, (x_axes, y_axes) = plt.subplots(2, 1, figsize=(3.2, 4.8))
        check_x = RadioButtons(x_axes, self.feature_names, self.feature_x)
        check_y = RadioButtons(y_axes, self.feature_names, self.feature_y)

        x_axes.set_title('green - min, red - max')

        check_x.on_clicked(self.select_x)
        check_y.on_clicked(self.select_y)
        plt.sca(self.plot_axes)
        self.select_x(self.feature_names[self.feature_x])


    def area(self, rect, minx, maxx, miny, maxy, color):
        x1 = max(rect.lower[self.feature_x], minx)
        x2 = min(rect.upper[self.feature_x], maxx)

        y1 = max(rect.lower[self.feature_y], miny)
        y2 = min(rect.upper[self.feature_y], maxy)

        if x1 < x2 and y1 < y2:
            plt.xlim([minx, maxx])
            plt.ylim([miny, maxy])

            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], alpha=0.5, color=color)

    def show_cut(self):
        plt.cla()
        ax = plt.gca()
        ax.set_xlabel(self.feature_names[self.feature_x])
        ax.set_ylabel(self.feature_names[self.feature_y])

        x = self.data[:, self.feature_x]
        y = self.data[:, self.feature_y]
        plt.scatter(x, y, s=5)

        [minx, maxx] = plt.xlim()
        [miny, maxy] = plt.ylim()

        self.area(self.min_rect, minx, maxx, miny, maxy, 'green')
        self.area(self.max_rect, minx, maxx, miny, maxy, 'red')
        plt.show()
