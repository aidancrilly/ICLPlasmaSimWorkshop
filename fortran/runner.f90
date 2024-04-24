program runner

    use lax
    implicit none
    real, dimension(:,:), allocatable :: r,l,s,psi
    integer :: ix,iy,it,Nt,new_unit
    integer(kind=8) :: count_rate,count_start,count_finish
    real :: dtmult,recorded_time
    character(len=255) :: iqtime
    character(len=255) :: filename
    logical :: IO

    delta = 0.01
    Nx = 200
    Ny = 300
    call initialise_lax()

    allocate(r(Nx,Ny),l(Nx,Ny),s(Nx,Ny),psi(Nx,Ny))

    dtmult = 0.5
    dt = dtmult*delta/maxval(c)

    call initialise_psi_gaussian(r,l,s,psi,10*delta)
    
    call SYSTEM_CLOCK(count=count_start,count_rate=count_rate)
    Nt = 1000
    IO = .false.
    do it = 1,Nt
        call lax_step(r,l,s,psi)
        if(mod(it,10) == 0 .and. IO) then
           write(iqtime,'(I)') it
           filename = './fortran/output/psi-'//trim(adjustl(iqtime))//'.txt'
           open(newunit=new_unit,file=trim(filename),action='write')
           do iy = 1,Ny
               write(new_unit,'(*(1pE15.5E3))') psi(:,iy)
           end do
           close(new_unit)
        end if
    end do
    call SYSTEM_CLOCK(count=count_finish)
	recorded_time = real(count_finish-count_start)/real(count_rate)
    print *, recorded_time
    
contains

    subroutine initialise_psi_gaussian(r,l,s,psi,sig)

        implicit none
        real, dimension(:,:) :: r,l,s,psi
        integer :: ix,iy
        real :: sig,x,y,r2

        do ix = 1,Nx
            x = delta*(ix-real(Nx)/2)
            do iy = 1,Ny
                y = delta*(iy-real(Ny)/2)
                r2 = x**2+y**2
                r(ix,iy) = -c(ix,iy)*x*exp(-0.5*r2/sig**2)/sig**2
                l(ix,iy) = -c(ix,iy)*y*exp(-0.5*r2/sig**2)/sig**2
                s(ix,iy) = 0.0
                psi(ix,iy) = exp(-0.5*r2/sig**2)
            end do
        end do

    end subroutine initialise_psi_gaussian

end program runner