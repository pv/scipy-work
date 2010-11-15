"""
Prototypes for algorithms needed for natural neighbour interpolation

"""


import numpy as np
from scipy.spatial import Delaunay

def addpoint(tri, x):
    """
    Add a new point to the given Delaunay triangulation.

    Returns
    -------
    neighbors : list of int
        Simplices whose each vertex the new point is connected to
        via faces.

    hull_faces : list of (ifacet, iface)
        Faces connecting the neighbors to each other.  The faces
        connecting the neighbors to the new point are not included.

    Notes
    -----

    The approach is the following. The Delaunay triangulation is a
    projection of the convex hull on a paraboloid in N+1 dimensions.
    Point P (with image P' on paraboloid) added inside the convex hull
    of the triangulation is connected to vertex V (image V') iff

        The ray between P' and V' does not intersect the convex hull

    This occurs iff

        The point P' is on the positive side of the hyperplane of the
        facet where V' belongs to.

    We find the set S(P) of such facets as follows:

        If P' is on the positive side, the facet in S(P),
        and zero or more of its neighbors may be in the set.

        The full set S(P) can be found via a 'flood-fill' algorithm.

    """
    x = np.asarray(x)

    horizon = []
    new_neighbors = []
    hull_faces = []
    seen = {}

    start = int(tri.find_simplex(x))
    assert start != -1
    seen[start] = True
    horizon.append((start, None))

    while horizon:
        facet, face = horizon.pop(0)
        seen[facet] = True
        if facet == -1:
            if face:
                hull_faces.append(face)
            continue
        dist = tri.plane_distance(x)[facet]
        if dist >= 0:
            new_neighbors.append(facet)
            for iface, neighbor in enumerate(tri.neighbors[facet]):
                if neighbor not in seen:
                    horizon.append((neighbor, (facet, iface)))
        else:
            if face:
                hull_faces.append(face)

    return new_neighbors, hull_faces

def getface(vertices, isimplex, iface):
    v = range(vertices.shape[1])
    v.remove(iface)
    return vertices[isimplex,v].tolist()

def _test_add(pts, x):
    pts2 = pts + [x]
    tri = Delaunay(pts)
    tri2 = Delaunay(pts2)

    def get_vert(tri, simp):
        vertices = set()
        for x in simp:
            vertices.update(tri.vertices[x])
        return list(sorted(vertices))

    expected = set((tri2.vertices == len(pts)).any(axis=1).nonzero()[0])

    neigh, faces = addpoint(tri, x)

    faces = [getface(tri.vertices, facet, iface) for facet, iface in faces]

    v1 = get_vert(tri2, expected)
    v2 = get_vert(tri, neigh) + [len(pts)]

    np.testing.assert_equal(v1, v2)

def test_add():
    pts = np.random.randn(40, 3).tolist()
    _test_add(pts, (0.5, 0.25, 0.75))

    pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]
    _test_add(pts, (0.5, 0.75))

def voronoi_center(tri, isimplex):
    """
    Compute the Voronoi center of a given simplex.

    Notes
    -----
    It can be found by solving::

      [ 1  y[0,0] ... y[0,n-1] ] [ (x**2).sum()-c**2  ] = [ -(y[0,:]**2).sum() ]
      [ .   .         .        ] [ -2 x[0]            ]   [ ...
      [ .   .         .        ] [ ...                ]   [ ...
      [ 1  y[n,0] ... y[n,n-1] ] [ -2 x[n-1]          ]   [ -(y[n,:]**2).sum() ]

    """
    ndim = tri.ndim

    y = tri.points[tri.vertices[isimplex]]
    y0 = y[0,:].copy()
    y -= y0

    lhs = np.zeros((ndim+1, ndim+1), float)
    lhs[:,0] = 1
    lhs[:,1:] = y

    rhs = np.zeros((ndim+1,), float)
    rhs[:] = -(y*y).sum(axis=1)

    return y0 - .5*np.linalg.solve(lhs, rhs)[1:]

def voronoi_centers(tri):
    centers = np.zeros((tri.nsimplex, tri.ndim), float)
    for isimplex in xrange(tri.nsimplex):
        centers[isimplex,:] = voronoi_center(tri, isimplex)
    return centers

def test_voronoi_centers_simple():
    pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]
    tri = Delaunay(pts)
    centers = voronoi_centers(tri)
    np.testing.assert_equal(centers, [(0, 0.5), (0.5, 1)])

def voronoi_volume(tri, ivertex):
    """
    Compute the volume of the Voronoi cell of a given vertex.

    """

    ind, iptr = tri.vertex_simplex
    ind_v, iptr_v = tri.vertex_neighbors

    simplices = iptr[ind[ivertex]:ind[ivertex+1]]
    neighbors = iptr_v[ind_v[ivertex]:ind_v[ivertex+1]]
    centers = voronoi_centers(tri)

    x0 = tri.points[ivertex]

    ndim_factorial = 1
    for j in xrange(tri.ndim):
        ndim_factorial *= (j+1)

    #
    # Compute the volume of the Voronoi polytope, using Cohen-Hickey
    # triangulation.
    #
    # J. Cohen and T. Hickey, J. Assoc. Comp. Mach. 26, 401 (1979).
    #

    # 1) Compute the (ndim-1)-faces F_k of the Polyhedron
    faces = []
    for v2 in neighbors:
        simp_list = []
        for s in simplices:
            if v2 in tri.vertices[s]:
                simp_list.append(s)
        faces.append(set(simp_list))

    # 2) Implement the algorithm

    volume = 0

    def vol(d, last, S):
        if d > 0:
            L = set([frozenset()])
            volume = 0
            for face in faces:
                z = last.intersection(face)
                if z not in L:
                    L.add(frozenset(z))
                    eta = sorted(list(z))[0]
                    if eta not in S:
                        S.add(eta)
                        volume += vol(d - 1, z, S)
                        S.remove(eta)
            return volume
        else:
            v = list(S.union(last))[1:]
            return abs(np.linalg.det(centers[v,:] - centers[simplices[0]]))/ndim_factorial

    last_0 = set(simplices)
    S_0 = set([simplices[0]])

    volume = vol(tri.ndim - 1, last_0, S_0)
    return volume

def factorial(x):
    fac = 1
    while x > 0:
        fac *= x
        x -= 1
    return fac

def unit_cube_center_voronoi_volume(d):
    return 2**d * (d/4.)**d / factorial(d)

def test_voronoi_volume_2d():
    pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    pts = np.array(pts, dtype=float)
    tri = Delaunay(pts)
    vol = voronoi_volume(tri, 4)

    expected_volume = unit_cube_center_voronoi_volume(2)
    np.testing.assert_allclose(vol, expected_volume)

def test_voronoi_volume_3d():
    pts = [[0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0],
           [0.0, 1.0, 0.0],
           [0.0, 1.0, 1.0],
           [1.0, 0.0, 0.0],
           [1.0, 0.0, 1.0],
           [1.0, 1.0, 0.0],
           [1.0, 1.0, 1.0],
           [0.5, 0.5, 0.5]]

    pts = np.array(pts, dtype=float)

    tri = Delaunay(pts)
    vol = voronoi_volume(tri, 8)

    expected_volume = unit_cube_center_voronoi_volume(3)
    np.testing.assert_allclose(vol, expected_volume)

def test_voronoi_volume_4d():
    pts = []
    for i in xrange(2):
        for j in xrange(2):
            for k in xrange(2):
                for l in xrange(2):
                    pts.append((i, j, k, l))
    pts.append([0.5, 0.5, 0.5, 0.5])

    tri = Delaunay(pts)
    vol = voronoi_volume(tri, 16)

    expected_volume = unit_cube_center_voronoi_volume(4)
    np.testing.assert_allclose(vol, expected_volume)

def test_voronoi_volume_5d():
    pts = []
    for i in xrange(2):
        for j in xrange(2):
            for k in xrange(2):
                for l in xrange(2):
                    for m in xrange(2):
                        pts.append((i, j, k, l, m))
    pts.append([0.5, 0.5, 0.5, 0.5, 0.5])

    tri = Delaunay(pts)
    vol = voronoi_volume(tri, 2**5)

    expected_volume = unit_cube_center_voronoi_volume(5)
    np.testing.assert_allclose(vol, expected_volume)

