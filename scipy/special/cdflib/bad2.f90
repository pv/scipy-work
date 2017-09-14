program main
  implicit none
  double precision :: adn, b, xx, yy, dnterm, dnterm_ex
  double precision :: alngam
  external :: alngam

  dnterm_ex = -232.27341231994683d0

  adn = 5.0000000000000001D+099
  b = 1
  xx = 1
  yy = 1.3333333333333333D-101
  dnterm = alngam(adn+b)-alngam(adn+1.0D0)-alngam(b)+&
       adn*log(xx)+b*log(yy)

  if (abs(dnterm - dnterm_ex).lt.1d-3) then
     write(*,*) 'OK'
  else
     write(*,*) dnterm, '!=', dnterm_ex
  end if
end program main
