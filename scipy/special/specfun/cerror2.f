c     Copyright (C)  Pauli Virtanen, 2010.
c     Distributed under the same New BSD license as Scipy.

      subroutine cerror2(z, w)
c
c     Compute error function w=erf(z) for a complex argument.
c
c     Parameters
c     ----------
c     z : complex*16, intent(in)
c         argument
c     w : complex*16, intent(out)
c         erf(z)
c
c     Notes
c     -----
c     Evaluates continued fraction expansions with Lenz's method.
c
      implicit none
      complex*16 z, w
      complex*16 z1, c, d, dw, a, b
      double precision pi
      integer k

      pi = 3.14159265358979323846264d0

      z1 = z
      if (dble(z) < 0) then
c        Inversion symmetry
         z1 = -z1
      end if
c
c     Lenz method for the continued fraction
c
c              1/2  2/2  3/2
c     w = z +  ---  ---  --- ...
c              z +  z +  z + 
c
c     for re(z) > 2 or abs(z) > 2*pi (the fraction does not 
c     converge well for small z)
c
c     and
c
c              -2 z**2  4 z**2  -6 z**2
c     w = 1 +  -------  ------  ------- ...
c              3    +   5   +   7    +
c
c     otherwise.
c
      if (abs(z) .gt. 2*pi .or. dble(z) .gt. 2d0) then
         w = z1
         c = w
         d = 0d0
         b = z
         do 10 k = 1, 120
            a = k / 2d0
            d = b + a * d
            if (d == 0) d = 1d-30
            c = b + a / c
            if (c == 0) c = 1d-30
            d = 1d0 / d
            dw = c*d
            w = w*dw
            write(*,*) k, cdabs(dw-1)
            if (cdabs(dw - 1) < 1d-15) goto 15
 10      continue
 15      w = 1d0 - cdexp(-z1*z1)/w/dsqrt(pi)
       else
         w = 1d0
         c = w
         d = 0d0
         do 20 k = 1, 120
            a = (-1)**k*2*k*z1*z1
            b = 2*k+1
            d = b + a * d
            if (d == 0) d = 1d-30
            c = b + a / c
            if (c == 0) c = 1d-30
            d = 1d0 / d
            dw = c*d
            w = w*dw
            write(*,*) k, cdabs(dw-1)
            if (cdabs(dw - 1) < 1d-15) goto 25
 20      continue
 25      w = 2*z1*cdexp(-z1*z1)/w/sqrt(pi)
      end if
      
      if (dble(z) < 0) then
         w = -w
      end if
      end
