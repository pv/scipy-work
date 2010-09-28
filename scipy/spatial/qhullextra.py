import numpy as np
from scipy.spatial import Delaunay

def addpoint(tri, x):
    """
    Add a new point to the given Delaunay triangulation.

    Returns
    -------
    new_neighbors : list of int
        Simplices whose each vertex the new point is connected to
        via ridges.

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
    seen = {}

    start = int(tri.find_simplex(x))
    assert start != -1
    seen[start] = True
    horizon.append(start)

    while horizon:
        facet = horizon.pop(0)
        if facet == -1:
            continue
        dist = tri.plane_distance(x)[facet]
        if dist >= 0:
            new_neighbors.append(facet)
            for neighbor in tri.neighbors[facet]:
                if neighbor not in seen:
                    seen[neighbor] = True
                    horizon.append(neighbor)

    return new_neighbors

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

    print get_vert(tri2, expected)
    print get_vert(tri, addpoint(tri, x)) + [len(pts)]



def test():
    pts = np.random.randn(40, 3).tolist()
    #pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]]
    test_add(pts,
             (0.25, 0.25, 0.25))
