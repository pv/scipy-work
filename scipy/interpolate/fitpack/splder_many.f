      subroutine splder_many(t,n,c,nc,k,nu,x,y,m,e,wrk,ier)
c  subroutine splder_many works similarly as splder, except that it
c  evaluates multiple b-splines at once.
c
c  calling sequence:
c     call splder_many(t,n,c,nc,k,nu,x,y,m,e,wrk,ier)
c
c  input parameters:
c    t    : array, size (n,), which contains the position of the knots.
c    n    : integer, giving the total number of knots of s(x).
c    nc   : integer, giving the number of coefficient sets
c    c    : array, size (n, nc), which contains the b-spline coefficients.
c    k    : integer, giving the degree of s(x).
c    nu   : integer, order of derivative
c    x    : array, length (m, nx), which contains the points where s(x) must
c           be evaluated.
c    m    : integer, giving the number of points where s(x) must be
c           evaluated.
c    e    : integer, if 0 the spline is extrapolated from the end
c           spans for points not in the support, if 1 the spline
c           evaluates to zero for those points, and if 2 ier is set to
c           1 and the subroutine returns.
c    wrk  : array, size(n), working space. not referenced if nu == 0
c
c  output parameter:
c    y    : array,length (m,nc), giving the value of s(x) at
c           the different points.
c    ier  : error flag
c      ier = 0 : normal return
c      ier = 1 : argument out of bounds and e == 2
c      ier =10 : invalid input data (see restrictions)
c
      implicit none
      double precision t, c, x, y, wrk
      integer n, nc, k, nu, m, e, ier
      dimension t(n)
      dimension x(m)
      dimension c(n, nc)
      dimension y(m, nc)
      dimension wrk(*)

      integer j

      ier = 0

      do 20 j = 1, nc
         if (nu.eq.0) then
            call splev(t,n,c(1,j),k,x,y(1,j),m,e,ier)
         else
            call splder(t,n,c(1,j),k,nu,x,y(1,j),m,e,
     *           wrk,ier)
         end if
         if (ier.ne.0) goto 30
 20   continue
 30   return
      end
