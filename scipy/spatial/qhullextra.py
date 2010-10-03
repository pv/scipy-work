import numpy as np
from scipy.spatial import Delaunay

def addpoint(tri, x):
    """
    Add a new point to the given Delaunay triangulation.

    Returns
    -------
    neighbors : list of int
        Simplices whose each vertex the new point is connected to
        via ridges.

    hull_ridges : list of (ifacet, iridge)
        Ridges connecting the neighbors to each other.  The ridges
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
    hull_ridges = []
    seen = {}

    start = int(tri.find_simplex(x))
    assert start != -1
    seen[start] = True
    horizon.append((start, None))

    while horizon:
        facet, ridge = horizon.pop(0)
        seen[facet] = True
        if facet == -1:
            if ridge:
                hull_ridges.append(ridge)
            continue
        dist = tri.plane_distance(x)[facet]
        if dist >= 0:
            new_neighbors.append(facet)
            for iridge, neighbor in enumerate(tri.neighbors[facet]):
                if neighbor not in seen:
                    horizon.append((neighbor, (facet, iridge)))
        else:
            if ridge:
                hull_ridges.append(ridge)

    return new_neighbors, hull_ridges

def getridge(vertices, isimplex, iridge):
    v = range(vertices.shape[1])
    v.remove(iridge)
    return vertices[isimplex,v].tolist()

def test_add(pts, x):
    pts2 = pts + [x]
    tri = Delaunay(pts)
    tri2 = Delaunay(pts2)

    def get_vert(tri, simp):
        vertices = set()
        for x in simp:
            vertices.update(tri.vertices[x])
        return list(sorted(vertices))

    expected = set((tri2.vertices == len(pts)).any(axis=1).nonzero()[0])

    neigh, ridges = addpoint(tri, x)

    ridges = [getridge(tri.vertices, facet, iridge) for facet, iridge in ridges]

    v1 = get_vert(tri2, expected)
    v2 = get_vert(tri, neigh) + [len(pts)]

    np.testing.assert_equal(v1, v2)
    print ridges

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

def test():
    #pts = np.random.randn(40, 3).tolist()
    pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]
    test_add(pts,
             (0.5, 0.75))
