! cd scipy/special/cdflib
! gfortran -Og -ggdb -o bad bad.f90 *.f
program main
  double precision :: f, dfn, dfd, pnonc, cum, ccum
  double precision :: cum_ex = 0.93550676527636045d0
  double precision :: ccum_ex = 6.4493234723639548d-2
  integer :: status

  f = 15
  dfn = 1d+100
  dfd = 2
  pnonc = 0.25
  cum = 8.3003028501329419d-322
  ccum = 6.9533558050453121d-310
  status = 32767

  call cumfnc(f, dfn, dfd, pnonc, cum, ccum, status)
  if (abs(cum-cum_ex).le.1d-8 .and. &
      abs(ccum-ccum_ex).le.1d-8 .and. &
      status.eq.0) then
    write(*,*) 'OK'
  else
    write(*,*) 'got:', cum, ccum, status
    write(*,*) 'expected:', cum_ex, ccum_ex, 0
  end if
end program main
