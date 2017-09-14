program main
  implicit none
  double precision :: adn, b, xx, yy, dnterm
  double precision :: alngam
  external :: alngam

  adn = 5.0000000000000001D+099
  b = 1
  xx = 1
  yy = 1.3333333333333333D-101
  dnterm = alngam(adn+b)-alngam(adn+1.0D0)-alngam(b)+&
       adn*log(xx)+b*log(yy)
  write(*,*) dnterm
end program main
