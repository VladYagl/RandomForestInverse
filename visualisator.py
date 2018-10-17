import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons


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

def visualise(value, data, names, feature_x=2, feature_y=3):
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

    print(x1, x2, y1, y2)

    plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], alpha=0.5, color='red')

    axes = plt.gca()
    rax = plt.axes([0, 0, 0.5, 0.5])
    check = CheckButtons(rax, names)

    def func(label):
        index = names.index(label)
        rax.clear()
        print("vis", index)
        visualise(value, data, names, index, feature_y)

    check.on_clicked(func)

    plt.sca(axes)
    plt.show()

