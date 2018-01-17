from __future__ import division

from fractions import Fraction as Frac
import numpy
import numpy.linalg
import matplotlib.pyplot as plt

N = 8 + 1

points_in_ek = 4

start_ek = [[-1.0, 0.0], [0.0, 0.0], [0.0, -1.0]]

ek_points = [[4, 5, 2, 1], [5, 6, 3, 2], [7, 8, 6, 5]]

center_of_edges_sort_by_ek_Neumann = [[[-1.0, 0.5], [-0.5, 1.0]], [[0.5, 1.0], [1.0, 0.5]], [[1.0, -0.5], [0.5, -1.0]]]


def mes():
    # make B matrix
    B = numpy.zeros((N, N))

    # make L matrix
    L = numpy.zeros(N)

    for k in range(0, 3):
        for i in range(0, 4):
            # add integrals to L matrix
            for val in center_of_edges_sort_by_ek_Neumann[k]:
                center_x = val[0]
                center_y = val[1]
                L[ek_points[k][i]] += g_func(center_x, center_y) * fi_func(center_x, center_y, i, k)

            # add integrals to B matrix
            # refator
            for j in range(0, 4):
                der_by_x = derivative_fi(i + 1, True) * derivative_fi(j + 1, True)
                der_by_y = derivative_fi(i + 1, False) * derivative_fi(j + 1, False)
                B[ek_points[k][i]][ek_points[k][j]] += der_by_x + der_by_y

    # boundary condition
    for i in range(0, N):
        B[4][i] = 0
        B[5][i] = 0
        B[7][i] = 0

    L[4] = 0
    L[5] = 0
    L[7] = 0

    B[4][4] = 1
    B[5][5] = 1
    B[7][7] = 1

    B = numpy.delete(B, 0, 0)
    B = numpy.delete(B, 0, 1)
    L = numpy.delete(L, 0, 0)

    #print(B)

    #print(L)

    a = numpy.linalg.solve(B, L)

    # Calculating values for plotting
    n = 100
    # sample
    xs = numpy.linspace(-1, 1, n)
    ys = numpy.linspace(-1, 1, n)
    # values
    Z = numpy.zeros((len(xs), len(ys)))

    for (i, x) in enumerate(xs):
        for (j, y) in enumerate(ys):
            # Pick Ek
            if y > 0:
                if x > 0:
                    k = 1
                else:
                    k = 0
            elif x > 0:
                k = 2
            else:
                continue  # u=0, skip
            # add each shape function
            for q in range(0, 4):
                Z[i, j] += a[ek_points[k][q] - 1] * fi_func(x, y, q, k)

    # Plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(30, -110)

    X, Y = numpy.meshgrid(xs, ys)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'))

    plt.show()


def fi_func(x, y, xi, ek):
    return {
        1: (1 - (x - start_ek[ek][0])) * (1 - (y - start_ek[ek][1])),
        2: (x - start_ek[ek][0]) * (1 - (y - start_ek[ek][1])),
        3: (x - start_ek[ek][0]) * (y - start_ek[ek][1]),
        4: (1 - (x - start_ek[ek][0])) * (y - start_ek[ek][1])
    }[xi + 1]


def g_func(x, y):
    return ((x + y) ** 2 / 2) ** Frac(1, 3)


def derivative_fi(xi, der_by_x):
    # der_by_x - true if derivative by x
    if xi == 3:
        return 0.5
    elif xi == 2 and der_by_x:
        return 0.5
    elif xi == 4 and not der_by_x:
        return 0.5
    else:
        return -0.5


mes()
