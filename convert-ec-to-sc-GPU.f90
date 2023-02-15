! Convert EC 
  module flow_data
   implicit none
   integer:: nx,ny,nz
   real*8,allocatable,dimension(:,:,:):: x3d, y3d, z3d
   real*8,allocatable,dimension(:,:)::  r2d,z2d
   real*8,allocatable,dimension(:,:,:):: d1,u1,v1,w1,T1            ! ec type
   real*8,allocatable,dimension(:,:,:):: d2,u2,v2,w2,T2            ! sc type
 end


 !--------------------------------------------------------------
  use flow_data
  implicit none
  integer:: NB,i,j,k,i0
  print*, "convert Mesh3d.dat & flow3d-plot3d.dat"
  open(99,file="Mesh3d-new.dat",form="unformatted")
  read(99) NB
  read(99) nx,ny,nz
  allocate(x3d(nx,ny,nz),y3d(nx,ny,nz),z3d(nx,ny,nz))
  allocate(d1(nx,ny,nz),u1(nx,ny,nz),v1(nx,ny,nz),w1(nx,ny,nz),T1(nx,ny,nz))

  read(99)     (((x3d(i,j,k),i=1,nx),j=1,ny),k=1,nz) ,  &
               (((y3d(i,j,k),i=1,nx),j=1,ny),k=1,nz) , &
               (((z3d(i,j,k),i=1,nx),j=1,ny),k=1,nz)

  close(99)
  print*, "read mesh OK"
  open(100,file="flow3d-plot3d-new.dat", form="unformatted")
  read(100)     (((d1(i,j,k),i=1,nx),j=1,ny),k=1,nz) ,  &
               (((u1(i,j,k),i=1,nx),j=1,ny),k=1,nz) ,  &
               (((v1(i,j,k),i=1,nx),j=1,ny),k=1,nz),   &
               (((w1(i,j,k),i=1,nx),j=1,ny),k=1,nz),   &
               (((T1(i,j,k),i=1,nx),j=1,ny),k=1,nz)
  close(100)

  print*, "read data OK"
  open(99,file="Mesh3d-new-GPU.dat",form="unformatted")
	call write3d(99,nx,ny,nz,x3d)
	call write3d(99,nx,ny,nz,y3d)	   
	call write3d(99,nx,ny,nz,z3d)	   
	close(99)

  open(99,file="flow3d-plot3d-new-GPU.dat",form="unformatted")
	call write3d(99,nx,ny,nz,d1)
	call write3d(99,nx,ny,nz,u1)	   
	call write3d(99,nx,ny,nz,v1)	
  call write3d(99,nx,ny,nz,w1)	   
	call write3d(99,nx,ny,nz,T1)   
	close(99)

  print*, "Write new data OK"

  deallocate(x3d,y3d,z3d,d1,u1,v1,w1,T1)
  end

!---------------------------------------------------------------
     subroutine write3d(no,nx,ny,nz,u3d)
	 implicit none
	 integer:: no,nx,ny,nz,k
	 real*8:: u3d(nx,ny,nz)
	 do k=1,nz
	 write(no) u3d(:,:,k)
	 enddo
	 end