cdef extern from "boost/math/special_functions.hpp":
     double cyl_bessel_jv "boost::math::cyl_bessel_j" (double, double) nogil
     double cyl_bessel_jn "boost::math::cyl_bessel_j" (int, double) nogil

cdef inline double boost_bessel_jv(double v, double x):
     return cyl_bessel_jv(v, x)

cdef inline double boost_bessel_jn(int n, double x):
     return cyl_bessel_jn(n, x)
