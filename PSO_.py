import os
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

os.chdir("~/PB3")


# =============================================================================
# helper functions
# =============================================================================
def obj_fn(x, y):
    """ Eggholder objective function value for input decision variables.
    Args:
        x (float): value for decision variable x.
        y (float): value for decision variable y.
    Returns:
        Objective function value given x and y args.
    """

    return -(y + 47)*math.sin(math.sqrt(abs(y + x/2 + 47))) - \
        x*math.sin(math.sqrt(abs(x - (y + 47))))


def rhos():
    """ Generate 2 random numbers from uniform distribution.
    Returns:
        rho1 (float): Random number between 0 (inclusive) and 1 (exclusive).
        rho2 (float): Random number between 0 (inclusive) and 1 (exclusive).
    """
    rho1, rho2 = np.random.uniform(0, 1), np.random.uniform(0, 1)

    return (rho1, rho2)


def eggholder_obj_space(pso_xs=[None], pso_ys=[None], pso_zs=[None]):
    """ Visualise eggholder function in 3D objective space.
    Args:
        pso_xs (float): final swarm x values.
        pso_ys (float): final swarm y values.
        pso_zs (float): final swarm z values.
    """
    rmin, rmax = -512, 512
    xax, yax = np.arange(rmin, rmax, 0.5), np.arange(rmin, rmax, 0.5)
    x, y = np.meshgrid(xax, yax)
    vect_obj_fn = np.vectorize(obj_fn)
    z = vect_obj_fn(x, y)

    figure = plt.figure()
    ax = figure.gca(projection="3d")
    ax.plot_surface(x, y, z, cmap="gist_rainbow", alpha=.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # add pso pts:
    try:
        for i in zip(pso_xs, pso_ys, pso_zs):
            ax.plot(i[0],
                    i[1],
                    i[2],
                    markerfacecolor="k",
                    markeredgecolor="k",
                    marker="o",
                    markersize=5,
                    alpha=.95)

    except ValueError:
        pass

    # plot global optimum as reference point
    ax.plot(512,
            404.2319,
            -959.6407,
            markerfacecolor="r",
            markeredgecolor="r",
            marker="o",
            markersize=5,
            alpha=.65)

    plt.show()


def xi_vi_pi_pg_init(n_particles):
    """ Initialisation of xv, vi, pi, and pg values according to PSO
    pseudocode.
    Args:
        n_particles (int): Number of particles to initialise for swarm.
    Returns:
        xi (list): location of current solution found for each of n_particles.
        vi (list): vx and vy component velocities for each of n_particles.
        pi (list): location of best solution found for each of n_particles.
        pg (tuple): best location in pi vector represented by an (x, y) pair.
    """
    rmin, rmax = -512, 513
    xi = [(np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax)) for i in range(n_particles)]

    vi = [(np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax)) for i in range(n_particles)]

    pi = copy.deepcopy(xi)  # inital best pos = initial pos

    pg = pg_get(xi)

    return (xi, vi, pi, pg)


def hl_xi_vi_pi_pg_init(n_particles):
    """ High-level pso initialisation of xv, vi, pi, and pg values according to
    PSO pseudocode.
    Args:
        n_particles (int): Number of particles to initialise for swarm.
    Returns:
        xi (list): location of current solution found for each of n_particles.
        vi (list): vx and vy component velocities for each of n_particles.
        pi (list): location of best solution found for each of n_particles.
        pg (tuple): best location in pi vector represented by (w, c1, c2).
    """
    rmin, rmax = 0.01, 10
    xi = [(np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax)) for i in range(n_particles)]

    vi = [(np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax),
           np.random.uniform(rmin, rmax)) for i in range(n_particles)]

    pi = copy.deepcopy(xi)  # inital best pos = initial pos

    pg = hl_pg_get(xi)

    return (xi, vi, pi, pg)


def pg_get(xi):
    """ Calculate pg given an xi vector for low-level pso.
    Args:
        xi (list): location of current solution found for each particle.
    Returns:
        pg (tuple): best location in pi vector according to obj_fn().
    """
    gbest_ind = np.argmin([obj_fn(loc[0], loc[1]) for loc in xi])
    pg = xi[gbest_ind]  # get lowest obj function value x/y pos

    return pg


def hl_pg_get(xi):
    """ Calculate pg given an xi vector for high-level pso.
    Args:
        xi (list): location of current solution found for each particle.
    Returns:
        pg (tuple): best location in pi vector according to low_level_pso()
        then obj_fn().
    """
    ll_pgs = [low_level_pso(param_tup[0],
                            param_tup[1],
                            param_tup[2],
                            plotting=False) for param_tup in xi]
    ll_objs = [obj_fn(*ll_pg) for ll_pg in ll_pgs]
    gbest_ind = np.argmin(ll_objs)
    pg = xi[gbest_ind]  # get lowest obj function value x/y pos

    return pg


def v_plus1_fn(x, v, p, pg, w, c1, c2):
    """ Generate updated velocity value according to PSO velcocity update
    equation. Return value used in vi_xi_plus1().
    Args:
        x (float): position at t.
        v (float): velocity at t.
        p (float): particle best position at t.
        pg (float): global best position at t.
        w (float): inertia.
        c1 (float): cognitive factor 1.
        c2 (float): cognitive factor 2.
    Returns:
        vi_plus1 (float): velocity value at t+1.
    """
    rho1, rho2 = rhos()
    vi_plus1 = w*v + rho1*c1*(p - x) + rho2*c2*(pg - x)

    return vi_plus1


def x_plus1_fn(x, v):
    """ Update position according to PSO position update equation.
    Args:
        x (float): position at t.
        v (float): velocity at t.
    Returns:
        xi_plus1 (float): position value at t+1.
    """
    xi_plus1 = x + v

    return xi_plus1


def vi_xi_plus1(xi, vi, pi, w, c1, c2):
    """ Generate updated velocity vector according to formula detailed in
    helper function for low-level pso.
    Args:
        xi (list): position vector at t.
        v (list): velocity vector at t.
        pi (list): particle best position vector at t.
        w (float): inertia.
        c1 (float): cognitive factor 1.
        c2 (float): cognitive factor 2.
    Returns:
        xi (list): position vector at t.
        vi_plus1 (list): velocity vector at t+1.
        xi_plus1 (list): position vector at t+1.
    """
    pg = pg_get(xi)
    # unzip
    xi_x, xi_y = [loc[0] for loc in xi], [loc[1] for loc in xi]
    vi_x, vi_y = [loc[0] for loc in vi], [loc[1] for loc in vi]
    pi_x, pi_y = [loc[0] for loc in pi], [loc[1] for loc in pi]

    # new vs
    vi_plus1_x = [v_plus1_fn(*tup, pg[0], w, c1, c2) for
                  tup in zip(xi_x, vi_x, pi_x)]
    vi_plus1_y = [v_plus1_fn(*tup, pg[1], w, c1, c2) for
                  tup in zip(xi_y, vi_y, pi_y)]
    vi_plus1 = [*zip(vi_plus1_x, vi_plus1_y)]

    # new xs
    xi_plus1_x = [x_plus1_fn(*tup) for tup in zip(xi_x, vi_plus1_x)]
    xi_plus1_y = [x_plus1_fn(*tup) for tup in zip(xi_y, vi_plus1_y)]
    xi_plus1 = [*zip(xi_plus1_x, xi_plus1_y)]

    return(xi, vi_plus1, xi_plus1)


def hl_vi_xi_plus1(xi, vi, pi, w, c1, c2):
    """ Generate updated velocity vector according to formula detailed in
    helper function for high-level pso.
    Args:
        x (float): position vector at t.
        v (float): velocity vector at t.
        pi (float): particle best position vector at t.
        w (float): inertia.
        c1 (float): cognitive factor 1.
        c2 (float): cognitive factor 2.
    Returns:
        xi (float): position vector at t.
        vi_plus1 (float): velocity vector at t+1.
        xi_plus1 (float): position vector at t+1.
    """
    pg = hl_pg_get(xi)
    # unzip
    xi_x, xi_y, xi_z = [param_tup[0] for param_tup in xi], \
                       [param_tup[1] for param_tup in xi], \
                       [param_tup[2] for param_tup in xi]

    vi_x, vi_y, vi_z = [param_tup[0] for param_tup in vi], \
                       [param_tup[1] for param_tup in vi], \
                       [param_tup[2] for param_tup in vi]

    pi_x, pi_y, pi_z = [param_tup[0] for param_tup in pi], \
                       [param_tup[1] for param_tup in pi], \
                       [param_tup[2] for param_tup in pi]
    # new vs
    vi_plus1_x = [v_plus1_fn(*tup, pg[0], w, c1, c2) for
                  tup in zip(xi_x, vi_x, pi_x)]
    vi_plus1_y = [v_plus1_fn(*tup, pg[1], w, c1, c2) for
                  tup in zip(xi_y, vi_y, pi_y)]
    vi_plus1_z = [v_plus1_fn(*tup, pg[2], w, c1, c2) for
                  tup in zip(xi_z, vi_z, pi_z)]
    vi_plus1 = [*zip(vi_plus1_x, vi_plus1_y, vi_plus1_z)]

    # new xs
    xi_plus1_x = [x_plus1_fn(*tup) for tup in zip(xi_x, vi_plus1_x)]
    xi_plus1_y = [x_plus1_fn(*tup) for tup in zip(xi_y, vi_plus1_y)]
    xi_plus1_z = [x_plus1_fn(*tup) for tup in zip(xi_z, vi_plus1_z)]
    xi_plus1 = [*zip(xi_plus1_x, xi_plus1_y, xi_plus1_z)]

    return(xi, vi_plus1, xi_plus1)


def scale(xi, vi_plus1, xi_plus1_unscaled):
    """ Returns scaled input xi according to 2D map for low-level pso.
    Args:
        xi (list): position vector at t.
        vi_plus1 (list): unscaled velocity vector at t+1.
        xi_plus1_unscaled (list): unscaled position vector at t+1.
    Returns:
        xi (list): position vector at t.
        xi_plus1_scaled (list): scaled position vector at t+1.
    """
    xlb, xub, ylb, yub = -512, 512, -512, 512
    vi_plus1_new = copy.deepcopy(vi_plus1)
    xi_plus1_scaled = copy.deepcopy(xi_plus1_unscaled)

    # cases:
    # 1) none --> leave
    # 2) only x / only y --> only use sx / sy depending on offender
    # 3) x and y --> calc sx and sy and use min to multiply against both vx/vy
    counter = 0
    for loc in xi_plus1_unscaled:
        locx, locy = loc[0], loc[1]
        if locx < xlb and locy > yub:  # case 3 - xlb and yub
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locx > xub and locy > yub:  # case 3 - xub and yub
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locx < xlb and locy < ylb:  # case 3 - xlb and ylb
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locx > xub and locy < ylb:  # case 3 - xub and ylb
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locx < xlb:  # case 2 - xlb
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            n_val0 = vi_plus1_new[counter][0] * sx
            n_val1 = vi_plus1_new[counter][1] * sx
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locx > xub:  # case 2 - xub
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            n_val0 = vi_plus1_new[counter][0] * sx
            n_val1 = vi_plus1_new[counter][1] * sx
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locy < ylb:  # case 2 - ylb
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * sy
            n_val1 = vi_plus1_new[counter][1] * sy
            vi_plus1_new[counter] = (n_val0, n_val1)

        elif locy > yub:  # case 2 - yub
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * sy
            n_val1 = vi_plus1_new[counter][1] * sy
            vi_plus1_new[counter] = (n_val0, n_val1)

        # case 1 (no scaling needed) - incr counter and continue loop
        counter += 1

    # now that velocities edited, move the particles, use 'vi_plus1_new'
    # unzip
    xi_x, xi_y = [loc[0] for loc in xi], [loc[1] for loc in xi]
    vi_x, vi_y = [loc[0] for loc in vi_plus1_new], \
                 [loc[1] for loc in vi_plus1_new]

    # calc new pos
    xi_plus1_scaled_x = [x_plus1_fn(*tup) for tup in zip(xi_x, vi_x)]
    xi_plus1_scaled_y = [x_plus1_fn(*tup) for tup in zip(xi_y, vi_y)]
    xi_plus1_scaled = [*zip(xi_plus1_scaled_x, xi_plus1_scaled_y)]

    return (xi, xi_plus1_scaled)


def hl_scale(xi, vi_plus1, xi_plus1_unscaled):
    """ Returns scaled input xi according to 3D map for high-level pso. The
    search space may be understood as a square/rectangular prism having its 3
    dimensios representing w, c1 and c2. Out of bounds areas fall into cases
    2-4. Case 4 has 3 violations (outside x, y and z bounds), case 3 has 2
    (outside 2 bounds) and case 2 has 1 (outside 1 bound). Case 1 is
    unviolated. These cases are further detailed in the comments of the code
    below.
    Args:
        xi (list): position vector at t.
        vi_plus1 (list): unscaled velocity vector at t+1.
        xi_plus1_unscaled (list): unscaled position vector at t+1.
    Returns:
        xi (list): position vector at t.
        xi_plus1_scaled (list): scaled position vector at t+1.
    """
    xlb, xub, ylb, yub, zlb, zub = 0.01, 10, 0.01, 10, 0.01, 10
    vi_plus1_new = copy.deepcopy(vi_plus1)
    xi_plus1_scaled = copy.deepcopy(xi_plus1_unscaled)

    # used top down approach in terms of complexity, allowing shorter code
    # otherwise, need to add that unmentioned dims are in normal range
    # (within bounds)
    counter = 0
    for param_tup in xi_plus1_unscaled:
        ptupx, ptupy, ptupz = param_tup[0], param_tup[1], param_tup[2]
        # case 4: 8x cases
        # 1
        if ptupx < xlb and ptupy > yub and ptupz < zlb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 2
        elif ptupx > xub and ptupy > yub and ptupz < zlb:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 3
        elif ptupx > xub and ptupy < ylb and ptupz < zlb:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 4
        elif ptupx < xlb and ptupy < ylb and ptupz < zlb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 5
        elif ptupx < xlb and ptupy > yub and ptupz > zub:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 6
        elif ptupx > xub and ptupy > yub and ptupz > zub:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 7
        elif ptupx > xub and ptupy < ylb and ptupz > zub:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 8
        elif ptupx < xlb and ptupy < ylb and ptupz > zub:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # case 3
        # x/y/z1, 2, 3, 4 (clockwise)
        # z1
        elif ptupx < xlb and ptupy > yub:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # z2
        elif ptupx > xub and ptupy > yub:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # z3
        elif ptupx > xub and ptupy < ylb:  # ISSUE! x6
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * abs(min(sx, sy))
            n_val1 = vi_plus1_new[counter][1] * abs(min(sx, sy))
            n_val2 = vi_plus1_new[counter][2] * abs(min(sx, sy))
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # z4
        elif ptupx < xlb and ptupy < ylb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sy)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sy)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sy)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # x1
        elif ptupy > yub and ptupz < zlb:
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # x2
        elif ptupy > yub and ptupz > zub:
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # x3
        elif ptupy < ylb and ptupz > zub:
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # x4
        elif ptupy < ylb and ptupz < zlb:
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            sz = (yub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sy, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sy, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sy, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # y1
        elif ptupx < xlb and ptupz < zlb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # y2
        elif ptupx < xlb and ptupz > zub:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sz = (yub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # y3
        elif ptupx > xub and ptupz > zub:  # ISSUE!
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            sz = (yub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * abs(min(sx, sz))
            n_val1 = vi_plus1_new[counter][1] * abs(min(sx, sz))
            n_val2 = vi_plus1_new[counter][2] * abs(min(sx, sz))
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # y4
        elif ptupx > xub and ptupz < zlb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            sz = (yub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * min(sx, sz)
            n_val1 = vi_plus1_new[counter][1] * min(sx, sz)
            n_val2 = vi_plus1_new[counter][2] * min(sx, sz)
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # case 2
        # 1
        elif ptupy > yub:
            sy = (yub - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * sy
            n_val1 = vi_plus1_new[counter][1] * sy
            n_val2 = vi_plus1_new[counter][2] * sy
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 2
        elif ptupx > xub:
            sx = (xub - xi[counter][0])/vi_plus1[counter][0]
            n_val0 = vi_plus1_new[counter][0] * sx
            n_val1 = vi_plus1_new[counter][1] * sx
            n_val2 = vi_plus1_new[counter][2] * sx
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 3
        elif ptupy < ylb:
            sy = (ylb - xi[counter][1])/vi_plus1[counter][1]
            n_val0 = vi_plus1_new[counter][0] * sy
            n_val1 = vi_plus1_new[counter][1] * sy
            n_val2 = vi_plus1_new[counter][2] * sy
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 4
        elif ptupx < xlb:
            sx = (xlb - xi[counter][0])/vi_plus1[counter][0]
            n_val0 = vi_plus1_new[counter][0] * sx
            n_val1 = vi_plus1_new[counter][1] * sx
            n_val2 = vi_plus1_new[counter][2] * sx
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 5
        elif ptupz > zub:
            sz = (zub - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * sz
            n_val1 = vi_plus1_new[counter][1] * sz
            n_val2 = vi_plus1_new[counter][2] * sz
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # 6
        elif ptupz < zlb:
            sz = (zlb - xi[counter][2])/vi_plus1[counter][2]
            n_val0 = vi_plus1_new[counter][0] * sz
            n_val1 = vi_plus1_new[counter][1] * sz
            n_val2 = vi_plus1_new[counter][2] * sz
            vi_plus1_new[counter] = (n_val0, n_val1, n_val2)

        # case 1
        counter += 1

    # now that velocities edited, move the particles, use 'vi_plus1_new'
    # unzip
    xi_x, xi_y, xi_z = [loc[0] for loc in xi], \
                       [loc[1] for loc in xi], \
                       [loc[2] for loc in xi]
    vi_x, vi_y, vi_z = [loc[0] for loc in vi_plus1_new], \
                       [loc[1] for loc in vi_plus1_new], \
                       [loc[2] for loc in vi_plus1_new]

    # calc new pos
    xi_plus1_scaled_x = [x_plus1_fn(*tup) for tup in zip(xi_x, vi_x)]
    xi_plus1_scaled_y = [x_plus1_fn(*tup) for tup in zip(xi_y, vi_y)]
    xi_plus1_scaled_z = [x_plus1_fn(*tup) for tup in zip(xi_z, vi_z)]
    xi_plus1_scaled = [*zip(xi_plus1_scaled_x,
                            xi_plus1_scaled_y,
                            xi_plus1_scaled_z)]

    return (xi, xi_plus1_scaled)


def pi_pg_update(xi, pi, pg, xi_plus1):
    """ Generate updated values for pi and pg for low-level pso.
    Args:
        xi (list): position vector at t.
        pi (list): particle best position vector at t.
        pg (tuple): best location in pi vector according to obj_fn()
        xi_plus1 (list): position vector at t+1.
    Returns:
        pi_old (list): initial particle best position vector.
        pg_old (tuple): initial best location in pi vector.
        pi (list): updated particle best position vector.
        pg (tuple): updated best location in pi vector.
    """
    pi_old = copy.deepcopy(pi)
    pg_old = copy.deepcopy(pg)

    # pi
    pi_candidates = xi_plus1
    pis_paired = [*zip(pi, pi_candidates)]
    pi_new = copy.deepcopy(pi)
    counter = 0
    for i in pis_paired:
        old, new = obj_fn(*i[0]), obj_fn(*i[1])
        if new < old:
            pi_new[counter] = (i[1][0], i[1][1])

        counter += 1

    # pg
    pg_candidate = pg_get(xi_plus1)
    if obj_fn(*pg_candidate) < obj_fn(*pg):
        pg = pg_candidate

    return (pi_old, pg_old, pi, pg)


def hl_pi_pg_update(xi, pi, pg, xi_plus1):
    """ Generate updated values for pi and pg for high-level pso.
    Args:
        xi (list): position vector at t.
        pi (list): particle best position vector at t.
        pg (tuple): best location in pi vector according to obj_fn()
        xi_plus1 (list): position vector at t+1.
    Returns:
        pi_old (list): initial particle best position vector.
        pg_old (tuple): initial best location in pi vector.
        pi (list): updated particle best position vector.
        pg (tuple): updated best location in pi vector.
    """
    pi_old = copy.deepcopy(pi)
    pg_old = copy.deepcopy(pg)

    # pi
    pi_candidates = xi_plus1
    pis_paired = [*zip(pi, pi_candidates)]
    pi_new = copy.deepcopy(pi)
    counter = 0
    for i in pis_paired:
        old, new = low_level_pso(*i[0]), low_level_pso(*i[1])
        if new < old:
            pi_new[counter] = (i[1][0], i[1][1], i[1][2])

        counter += 1

    # pg
    pg_candidate = hl_pg_get(xi_plus1)
    if low_level_pso(*pg_candidate) < low_level_pso(*pg):
        pg = pg_candidate

    return (pi_old, pg_old, pi, pg)


# =============================================================================
# main functions
# =============================================================================
def high_level_pso(w=1, c1=1, c2=1, n_particles=10):
    """ Discover behavioural parameters of optimiser.
    Args:
        w (float): inertia.
        c1 (float): cognitive factor 1.
        c2 (float): cognitive factor 2.
        n_particles (int): number of particles to initialise for swarm.
    Returns:
        pg (tuple): tuple consisting of hyperparameter search results (for low-
        level PSO).
            w (float): inertia.
            c1 (float): cognitive factor 1.
            c2 (float): cognitive factor 2.
    """
    # initialise
    xi, vi, pi, pg = hl_xi_vi_pi_pg_init(n_particles)
    # first iteration - move
    xi, vi_plus1, xi_plus1 = hl_vi_xi_plus1(xi, vi, pi, w, c1, c2)
    # scale
    xi, xi_plus1 = hl_scale(xi, vi_plus1, xi_plus1)
    # update
    _, pg_old, pi, pg = hl_pi_pg_update(xi, pi, pg, xi_plus1)
    for i in range(1):
        print(i)
        xi, vi_plus1, xi_plus1 = hl_vi_xi_plus1(xi, vi_plus1, pi, w, c1, c2)
        xi, xi_plus1 = hl_scale(xi, vi_plus1, xi_plus1)
        _, pg_old, pi, pg = hl_pi_pg_update(xi, pi, pg, xi_plus1)

    return pg


def low_level_pso(w, c1, c2, n_particles=50, plotting=False):
    """ Discover solutions to actual problem.
    Args:
        w (float): inertia.
        c1 (float): cognitive factor 1.
        c2 (float): cognitive factor 2.
        n_particles (int): number of particles to initialise for swarm.
    Returns:
        pg (tuple): tuple consisting of pso search results (solution to
        eggholder minimisation).
            x (float): value for decision variable x.
            y (float): value for decision variable y.
    """
    # initialise
    xi, vi, pi, pg = xi_vi_pi_pg_init(n_particles)
    # first iteration - move
    xi, vi_plus1, xi_plus1 = vi_xi_plus1(xi, vi, pi, w, c1, c2)
    # scale
    xi, xi_plus1 = scale(xi, vi_plus1, xi_plus1)
    # update
    _, pg_old, pi, pg = pi_pg_update(xi, pi, pg, xi_plus1)
    # print(obj_fn(*pg))
    no_impr_count = 0
    # until stopping condition...
    while True:
        # move
        xi, vi_plus1, xi_plus1 = vi_xi_plus1(xi, vi_plus1, pi, w, c1, c2)
        # scale
        xi, xi_plus1 = scale(xi, vi_plus1, xi_plus1)
        # update
        _, pg_old, pi, pg = pi_pg_update(xi, pi, pg, xi_plus1)
        # print(obj_fn(*pg))
        if pg_old == pg:
            no_impr_count += 1

        else:
            no_impr_count = 0

        # print(no_impr_count)
        if no_impr_count == 10:
            break

    # plot before end
    if plotting:
        xs = [i[0] for i in xi_plus1]
        ys = [i[1] for i in xi_plus1]
        zs = [obj_fn(*i) for i in xi_plus1]
        eggholder_obj_space(pso_xs=xs, pso_ys=ys, pso_zs=zs)

    return pg


if __name__ == "__main__":
    start = time.time()

    # run hybrid metaheuristic
    params = high_level_pso()
    pg = low_level_pso(*params, plotting=True)

    # write results to a log file
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
    with open(f"param_log/{now}.txt", "w", encoding="utf-8") as file:
        file.write(f"{params}\n")
        file.write(f"{pg}\n")

    # print results
    print(params)
    print(pg)

    end = time.time()
    # print elapsed time to run
    print(end - start)
