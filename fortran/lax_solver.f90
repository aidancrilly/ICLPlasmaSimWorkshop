module lax
    implicit none
    real    :: dt,delta
    real, dimension(:,:), allocatable :: F_x,F_y,c
    real, dimension(:,:), allocatable :: r_1,l_1,s_1,s_0
    integer :: Nx,Ny

contains

    subroutine initialise_lax()

        implicit none

        allocate(F_x(0:Nx+1,0:Ny+1),F_y(0:Nx+1,0:Ny+1),c(0:Nx+1,0:Ny+1))
        allocate(r_1(0:Nx+1,0:Ny+1),l_1(0:Nx+1,0:Ny+1))
        allocate(s_0(0:Nx+1,0:Ny+1),s_1(0:Nx+1,0:Ny+1))
        
        c(0:Nx+1,0:Ny+1) = 1.0

    end subroutine
    
    subroutine get_ghost(f_ghost,f)

        implicit none
        real, dimension(0:,0:) :: f_ghost
        real, dimension(:,:)   :: f
        integer :: ix,iy

        f_ghost(1:Nx,1:Ny) = f

        f_ghost(0,0)       = f(1,1)
        f_ghost(Nx+1,0)    = f(Nx,1)
        f_ghost(0,Ny+1)    = f(1,Ny)
        f_ghost(Nx+1,Ny+1) = f(Nx,Ny)

        ix = 0
        do iy = 1,Ny
            f_ghost(ix,iy) = f(1,iy)
        end do

        iy = 0
        do ix = 1,Nx
            f_ghost(ix,iy) = f(ix,1)
        end do

        ix = Nx+1
        do iy = 1,Ny
            f_ghost(ix,iy) = f(Nx,iy)
        end do

        iy = Ny+1
        do ix = 1,Nx
            f_ghost(ix,iy) = f(ix,Ny)
        end do

    end subroutine

    subroutine lax_update(f,jx,jy)

        implicit none
        real, dimension(0:,0:) :: f,jx,jy
        real, dimension(:,:), allocatable :: f_next
        integer :: ix,iy
        
        allocate(f_next(0:Nx+1,0:Ny+1))
        
        do ix = 1,Nx
            do iy = 1,Ny
                f_next(ix,iy) = 0.25 * (f(ix-1,iy) + f(ix+1,iy) + f(ix,iy-1) + f(ix,iy+1))
                f_next(ix,iy) = f_next(ix,iy) - 0.5 * (jx(ix+1,iy) - jx(ix-1,iy) + jy(ix,iy+1) - jy(ix,iy-1))
            end do
        end do
        
        f = f_next

    end subroutine lax_update

    subroutine lax_step(r,l,s,psi)

        implicit none
        real, dimension(:,:) :: r,l,s,psi
        integer :: ix,iy

        call get_ghost(s_0,s)
        call get_ghost(s_1,s)
        call get_ghost(r_1,r)
        call get_ghost(l_1,l)
        
        do ix = 0,Nx+1
            do iy = 0,Ny+1
                F_x(ix,iy) = -dt/delta*c(ix,iy)*r_1(ix,iy)
                F_y(ix,iy) = -dt/delta*c(ix,iy)*l_1(ix,iy)
            end do
        end do
        call lax_update(s_1,F_x,F_y)
        
        do ix = 0,Nx+1
            do iy = 0,Ny+1
                F_x(ix,iy) = -dt/delta*c(ix,iy)*s_0(ix,iy)
                F_y(ix,iy) = 0.0
            end do
        end do
        call lax_update(r_1,F_x,F_y)
        ! Reflective boundaries
        r_1(1,:)  = 0.0
        r_1(Nx,:) = 0.0

        do ix = 0,Nx+1
            do iy = 0,Ny+1
                F_x(ix,iy) = 0.0
                F_y(ix,iy) = -dt/delta*c(ix,iy)*s_0(ix,iy)
            end do
        end do
        call lax_update(l_1,F_x,F_y)
        ! Reflective boundaries
        l_1(:,1)  = 0.0
        l_1(:,Ny) = 0.0

        do ix = 1,Nx
            do iy = 1,Ny
                r(ix,iy) = r_1(ix,iy)
                l(ix,iy) = l_1(ix,iy)
                s(ix,iy) = s_1(ix,iy)
                psi(ix,iy) = psi(ix,iy) + 0.5 * dt * (s_0(ix,iy) + s_1(ix,iy))
            end do
        end do

    end subroutine lax_step

end module lax