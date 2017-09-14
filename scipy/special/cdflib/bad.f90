program main
  double precision :: f, dfn, dfd, pnonc, cum, ccum
  integer :: status

  f = 15
  dfn = 1d+100
  dfd = 2
  pnonc = 0.25
  cum = 8.3003028501329419d-322
  ccum = 6.9533558050453121d-310
  status = 32767

  call cumfnc(f, dfn, dfd, pnonc, cum, ccum, status)
  write(*,*) cum, ccum, status
end program main
