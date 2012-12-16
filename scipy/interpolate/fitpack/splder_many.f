      subroutine splder_many(t,nt,n,c,nc,k,nu,x,nx,y,m,e,wrk,ier)
c  subroutine splder_many works similarly as splder, except that it
c  evaluates multiple b-splines at once.
c
c  the parameters nt, nc, nx must either be 1, or equal to each other.
c  a value of 1 is treated specially and 'broadcasted' across the
c  others.
c
c  calling sequence:
c     call splder_many(t,nt,n,c,nc,k,nu,x,nx,y,m,e,wrk,ier)
c
c  input parameters:
c    t    : array, size (n, nt), which contains the position of the knots.
c    n    : integer, giving the total number of knots of s(x).
c    nt   : integer, giving the number of knot point sets
c    nc   : integer, giving the number of coefficient sets
c    c    : array, size (n, nc), which contains the b-spline coefficients.
c    k    : integer, giving the degree of s(x).
c    nu   : integer, order of derivative
c    x    : array, length (m, nx), which contains the points where s(x) must
c           be evaluated.
c    nx   : number of point sets where s(x) must be evaluated
c    m    : integer, giving the number of points where s(x) must be
c           evaluated.
c    e    : integer, if 0 the spline is extrapolated from the end
c           spans for points not in the support, if 1 the spline
c           evaluates to zero for those points, and if 2 ier is set to
c           1 and the subroutine returns.
c    wrk  : array, size(n), working space. not referenced if nu == 0
c
c  output parameter:
c    y    : array,length (m,max(nt,nc,nx)), giving the value of s(x) at
c           the different points.
c    ier  : error flag
c      ier = 0 : normal return
c      ier = 1 : argument out of bounds and e == 2
c      ier =10 : invalid input data (see restrictions)
c
      implicit none
      double precision t, c, x, y, wrk
      integer nt, n, nc, k, nu, nx, m, e, ier
      dimension t(n, nt)
      dimension x(m, nx)
      dimension c(n, nc)
      dimension y(m, *)
      dimension wrk(*)
      
      integer it, ic, ix, st, sc, sx, nmax, j

      write(*,*) 'nt', nt
      write(*,*) 'nc', nc
      write(*,*) 'nx', nx

      if (nt.eq.1) then
         st = 0
      else
         st = 1
      end if
      if (nc.eq.1) then
         sc = 0
      else
         sc = 1
      end if
      if (nx.eq.1) then
         sx = 0
      else
         sx = 1
      end if
      
      nmax = max(nx, max(nt, nc))

      it = 1
      ix = 1
      ic = 1
      do 20 j = 1, nmax
         if (nu.eq.0) then
            call splev(t(1,it),n,c(1,ic),k,x(1,ix),y(1,j),m,e,ier)
         else
            call splder(t(1,it),n,c(1,ic),k,nu,x(1,ix),y(1,j),m,e,
     *           wrk,ier)
         end if
         if (ier.ne.0) goto 30
         it = it + st
         ic = ic + sc
         ix = ix + sx
 20   continue
 30   return
      end
