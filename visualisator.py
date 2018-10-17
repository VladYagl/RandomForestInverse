import matplotlib.pyplot as plt


# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label='linear')
# plt.plot(x, x**2, label='quadratic')
# plt.plot(x, x**3, label='cubic')
#
# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title("Simple Plot")
#
# plt.legend()
#
# plt.show()

def visualise(value, data, feature_x=2, feature_y=3):
    print("feature_x", feature_x, "feature_y", feature_y)
    plt.cla()

    x = data[:, feature_x]
    y = data[:, feature_y]

    plt.scatter(x, y, s=5)

    [minx, maxx] = plt.xlim()
    [miny, maxy] = plt.ylim()

    x1 = max(value.lower[feature_x], minx)
    x2 = min(value.upper[feature_x], maxx)

    y1 = max(value.lower[feature_y], miny)
    y2 = min(value.upper[feature_y], maxy)

    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])

    plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], alpha=0.5, color='red')
    plt.show()

