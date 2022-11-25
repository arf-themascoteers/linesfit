import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import NamedTuple, List


def norm(arr):
    return arr / np.linalg.norm(arr)


def rotate(p, theta):
    cs = np.cos(theta)
    sn = np.sin(theta)

    x, y = p

    x2 = x * cs - y * sn
    y2 = x * sn + y * cs
    return np.array([x2, y2])


def rotateLeft(p):
    return rotate(p, np.pi / 2.0)


class Point(NamedTuple):
    x: float
    y: float


# An input line will have points randomly generated around it, the line is specified as two points
class InputLine(NamedTuple):
    start: np.array
    end: np.array
    points: np.array


def make_input_line(start: Point, end: Point, steps=20, spread=0.12) -> InputLine:
    seed = abs(int(10 * (start.x - 2 * end.x + 3 * start.y - 4 * end.y)))  # randomish but stable
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)

    r = np.random.RandomState(seed)
    dirForward = norm(end - start)
    dirLeft = rotateLeft(dirForward)
    points = []
    for i in range(steps):
        point = start + dirForward * float(i) / float(steps) + dirLeft * (r.rand() - 0.5) * spread
        points.append(point)

    return InputLine(start, end, points)


def make_random_line(r) -> InputLine:
    p1 = Point(x=r.rand(), y=r.rand())
    p2 = Point(x=r.rand(), y=r.rand())
    while (np.linalg.norm(np.array(p1) - np.array(p2)) < 0.5):
        p1 = Point(x=r.rand(), y=r.rand())
        p2 = Point(x=r.rand(), y=r.rand())
    return make_input_line(p1, p2)


# An output line is the two parameters of the y = mx + b equation (m=slope, b=y-intercept)
class OutputLine(NamedTuple):
    m: float
    b: float

    def y(self, x: float) -> float:
        return self.m * x + self.b

def runOne(solve, line1, line2, ax):
    lines = [line1, line2]

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)

    for line in lines:
        x, y = zip(*line.points)
        ax.scatter(x, y, c='g', s=15, edgecolors='none')

    # Run the solver
    all_points = np.array([
        p
        for line in lines
        for p in line.points
    ])
    output_lines = solve(all_points)

    # Plot the output lines
    for line in output_lines:
        xs = [0, 1]
        ys = [line.y(x) for x in xs]
        ax.plot(xs, ys, c='b', alpha=0.5)

    for line in lines:
        x, y = zip(*line.points)
        xs = [line.start[0], line.end[0]]
        ys = [line.start[1], line.end[1]]
        ax.plot(xs, ys, c='gray', alpha=0.5)


def run(num, solve):
    plt.rcParams["figure.figsize"] = (20, num * 5)

    r = np.random.RandomState(0)
    for i in range(num):
        ax = plt.subplot((num + 1) // 2, 2, i + 1, label=str(i))

        line1 = make_random_line(r)
        line2 = make_random_line(r)
        runOne(solve, line1, line2, ax)
    plt.show()


def solve(points: np.array) -> List[OutputLine]:
    r = np.random.RandomState(0)
    m1 = r.rand()
    b1 = r.rand()
    m2 = -m1
    b2 = r.rand()

    for i in range(10000):
        loss, m1, b1, m2, b2 = optimise(points,m1, b1, m2, b2)
        print(i,":",loss)

    line1 = OutputLine(m=m1, b=b1)
    line2 = OutputLine(m=m2, b=b2)
    return [line1, line2]

def optimise(points, m1, b1, m2, b2):
    loss = 0
    lr = 0.001
    grad_m1 = 0
    grad_b1 = 0
    grad_m2 = 0
    grad_b2 = 0

    for point in points:
        dist1 = m1*point[0] - point[1] + b1
        dist2 = m2*point[0] - point[1] + b2

        loss_dist1 = dist1 ** 2
        loss_dist2 = dist2 ** 2

        loss_for_this = 0

        if loss_dist1 < loss_dist2:
            loss_for_this = loss_dist1
            grad_dist1 = 2 * dist1
            grad_m1_for_this = grad_dist1 * point[0]
            grad_b1_for_this = grad_dist1
            grad_m1 = grad_m1 + grad_m1_for_this
            grad_b1 = grad_b1 + grad_b1_for_this
        else:
            loss_for_this = loss_dist2
            grad_dist2 = 2 * dist2
            grad_m2_for_this = grad_dist2 * point[0]
            grad_b2_for_this = grad_dist2
            grad_m2 = grad_m2 + grad_m2_for_this
            grad_b2 = grad_b2 + grad_b2_for_this

        loss = loss + loss_for_this

    descent_grad_m1 = lr * grad_m1
    descent_grad_b1 = lr * grad_b1
    descent_grad_m2 = lr * grad_m2
    descent_grad_b2 = lr * grad_b2

    m1 = m1 - descent_grad_m1
    b1 = b1 - descent_grad_b1
    m2 = m2 - descent_grad_m2
    b2 = b2 - descent_grad_b2

    return loss, m1, b1, m2, b2

run(4, solve)
