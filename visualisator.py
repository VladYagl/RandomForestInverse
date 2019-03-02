import matplotlib.pyplot as plt


def area(rect, feature_x, feature_y, minx, maxx, miny, maxy, color):
    x1 = max(rect.lower[feature_x], minx)
    x2 = min(rect.upper[feature_x], maxx)

    y1 = max(rect.lower[feature_y], miny)
    y2 = min(rect.upper[feature_y], maxy)

    if x1 < x2 and y1 < y2:
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])

        plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], alpha=0.5, color=color)


def visualise(min_rect, max_rect, data, feature_x=2, feature_y=3, feature_names=[]):
    print("feature_x", feature_x, "feature_y", feature_y)
    plt.cla()
    ax = plt.gca()
    ax.set_xlabel(feature_names[feature_x])
    ax.set_ylabel(feature_names[feature_y])

    x = data[:, feature_x]
    y = data[:, feature_y]
    plt.scatter(x, y, s=5)

    [minx, maxx] = plt.xlim()
    [miny, maxy] = plt.ylim()

    area(min_rect, feature_x, feature_y, minx, maxx, miny, maxy, 'green')
    area(max_rect, feature_x, feature_y, minx, maxx, miny, maxy, 'red')
    plt.show()
