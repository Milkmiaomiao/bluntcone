#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#include "mpi.h"
#include "pthread.h"

#define PI 3.141592653589793

FILE *fp;
FILE *fp1;
FILE *fp2;

MPI_Status status;

char str[2000];
char Mesh_old[50], Data_old[50], Mesh_new[50];
int init[3];
int my_id, n_processe;
int nx, ny, nz, NZ, *NPZ, *NP, *head, *NPE;//, NP_in[n_processe], NPZ_in[n_processe];
int NZ1, *NPZ1, *NP1, *NPE1;
int nx1,ny1,nz1,Iflag_half;
double *x3d, *y3d, *z3d;
double *x3d_in, *y3d_in, *z3d_in;
double *d_in, *u_in, *v_in, *w_in, *T_in;
double *dn, *un, *vn, *wn, *Tn;
double *x3dr, *y3dr, *z3dr;
double *d13d, *u13d, *v13d, *w13d, *T13d;
double *dnr, *unr, *vnr, *wnr, *Tnr;
double *x2, *y2, *z2, *d2, *u2, *v2, *w2, *T2;
double *x1, *y11, *z1, *d1, *u1, *v1, *w1, *T1; 
/*           ||
             ||      奇怪的报错
            \ /
   error: ‘y1’ redeclared as different kind of symbol
   double *x1, *y1, *z1, *d1, *u1, *v1, *w1, *T1;
                ^~
  In file included from /usr/include/features.h:375:0,
                   from /usr/include/stdio.h:27,
                   from bluntcon-flow3d0.c:1:
  /usr/include/bits/mathcalls.h:242:1: note: previous declaration of ‘y1’ was here
  __MATHCALL (y1,, (_Mdouble_));
*/


void mpi_init(int *Argc, char ***Argv);
void Read_parameter();
void Data_malloc();
void Read_oldmesh();
void Read_olddata();
void Read_olddata1();
void Read_newmesh();
void cz3d_to_2d_inlet();
void cz2d_to_2d(double (*Uy0)[ny], double (*Uz0)[ny], double (*Ud0)[ny], double (*Uu0)[ny], double (*Uv0)[ny], double (*Uw0)[ny], double (*UT0)[ny]);
double cz2d_seta(int j, double Us1, double Uf1[6], double (*Ur0)[ny], double (*Us0)[ny] ,double (*Ud0)[ny], double (*Uu0)[ny], double (*Uv0)[ny], double (*Uw0)[ny], double (*UT0)[ny]);
double * inter1d_6th(int m, double UUs1, double UUf1[m], int nx, double Useta0[nx], double Uf0[m][nx]);
int maxint(int a, int b);
int minint(int a, int b);
void write_inletsection();
void cz3d_to_2d();
void find_nearest_ijk(int (*Uist)[nx1], int (*Ujst)[nx1], int (*Ukst)[nx1]);
void write_outboundary();
void output_outflow();
void write_3d_init_file();

void Finalize();

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();
    
    Data_malloc();

    Read_oldmesh();

    Read_olddata();

    //Read_olddata1();
   
    Read_newmesh();

    cz3d_to_2d_inlet();

    write_inletsection();

    cz3d_to_2d();

    write_outboundary();

    output_outflow();

    write_3d_init_file();

    Finalize();

    return 0;
}

void mpi_init(int *Argc , char *** Argv){

	MPI_Init(Argc, Argv);

    MPI_Comm_rank(MPI_COMM_WORLD , &my_id);
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_processe);

}

void Read_parameter(){
    if(my_id == 0){
        if((fp = fopen("set-boundary.in", "r")) == NULL){
            printf("Can't open this file: 'set-boundary.in'\n");
            exit(0);
        }

        fgets(str, 2000, fp);
        fscanf(fp, "%d%d%d%d%d%d%d\n", &nx,&ny,&nz,&nx1,&ny1,&nz1,&Iflag_half);
        printf("Computation start...\nnx is %d\nny is %d\nnz is %d\nnx1 is %d\nny1 is %d\nnz1 is %d\nIflag_half is %d\n",
              nx, ny, nz, nx1, ny1, nz1, Iflag_half);
        
        /*fgets(str, 2000, fp);
        
        fscanf(fp, "%s%s%s\n", Mesh_old, Data_old, Mesh_new);
        printf("Mesh_old is %s\nData_old is %s\nMesh_new is %s\n",
              Mesh_old, Data_old, Mesh_new);*/

        fclose(fp);
    }
//-----------------------------------------------------------------------------------------------------
    int tmp1[7];
    char tmp2,tmp3,tmp4;

    if(my_id == 0){
        tmp1[0] = nx;
        tmp1[1] = ny;
        tmp1[2] = nz;
        tmp1[3] = nx1;
        tmp1[4] = ny1;
        tmp1[5] = nz1;
        tmp1[6] = Iflag_half;

    }

    MPI_Bcast(tmp1, 7, MPI_INT, 0, MPI_COMM_WORLD);


    if(my_id != 0){
        nx = tmp1[0];
        ny = tmp1[1];
        nz = tmp1[2];
        nx1 = tmp1[3];
        ny1 = tmp1[4];
        nz1 = tmp1[5];
        Iflag_half = tmp1[6];

    }
 //-------------------------------------------------------------------------------------------

    NZ = nz/n_processe;      

    if(my_id < nz%n_processe) NZ += 1; //每一个my_id中NZ取值都不相同

    NPZ = (int*)malloc(n_processe * sizeof(int));
    NP = (int*)malloc(n_processe * sizeof(int));
    NPE = (int*)malloc(n_processe * sizeof(int));

    memset((void*)NPZ, 0, n_processe*sizeof(int));   /*为malloc新申请的连续内存进行初始化*/
    memset((void*)NP, 0, n_processe*sizeof(int));
    memset((void*)NPE, 0, n_processe*sizeof(int));

    //int NP_in[n_processe], NPZ_in[n_processe];

    for(int i = 0; i < n_processe; i++){
        if(i < nz%n_processe){
            NPZ[i] = (int)nz/n_processe + 1;
            
        }else{
            NPZ[i] = (int)nz/n_processe;
        }
        //NPZ_in[i] = NPZ[i]*ny;
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];//偏移 
        //NP_in[i] = NP[i]*ny;  
        NPE[0] = NPZ[0];
        if(i != 0) NPE[i] = NPE[i-1] + NPZ[i];
    }

    if(NP[n_processe-1] != nz-NPZ[n_processe-1]) printf("NP is wrong![debug]\n");
 //============================ 为读取新网格====================================================
    NZ1 = nz1/n_processe;
    if(my_id < nz1%n_processe) NZ1 += 1;
    NPZ1 = (int*)malloc(n_processe * sizeof(int));
    NP1 = (int*)malloc(n_processe * sizeof(int));
    NPE1 = (int*)malloc(n_processe * sizeof(int));
    memset((void*)NPZ1, 0, n_processe*sizeof(int));   /*为malloc新申请的连续内存进行初始化*/
    memset((void*)NP1, 0, n_processe*sizeof(int));
    memset((void*)NPE1, 0, n_processe*sizeof(int));

    for(int i = 0; i < n_processe; i++){
        if(i < nz1%n_processe){
            NPZ1[i] = (int)nz1/n_processe + 1;
        }else{
            NPZ1[i] = (int)nz1/n_processe;
        }
        NP1[0] = 0;
        if(i != 0) NP1[i] = NP1[i-1] + NPZ1[i-1];//偏移    
        NPE1[0] = NPZ1[0];
        if(i != 0) NPE1[i] = NPE1[i-1] + NPZ1[i];
    }

}

#define Malloc_Judge(Memory)\
    {   if(Memory == NULL){\
        printf("Memory allocate error ! Can not allocate enough memory !!!\n");\
        exit(0); }\
    }


void Data_malloc(){
    x3d = (double*)malloc(nx * (ny) * NZ * sizeof(double));
    Malloc_Judge(x3d);

    y3d = (double*)malloc(nx * (ny) * NZ * sizeof(double));
    Malloc_Judge(y3d);

    z3d = (double*)malloc(nx * (ny) * NZ * sizeof(double));
    Malloc_Judge(z3d);  

    x3dr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(x3dr);

    y3dr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(y3dr);

    z3dr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(z3dr);  

    /*x3d_in = (double*)malloc(nx * ny * nz * sizeof(double));
    Malloc_Judge(x3d_in);*/

    y3d_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(y3d_in);

    z3d_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(z3d_in);

    d_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(d_in);

    u_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(u_in);

    v_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(v_in);

    w_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(w_in);

    T_in = (double*)malloc( ny * nz * sizeof(double));
    Malloc_Judge(T_in);


    head = (int*)malloc(5 * sizeof(int));

    dn = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(dn);

    un = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(un);

    vn = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(vn);

    wn = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(wn); 

    Tn = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(Tn);

    dnr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(dnr);

    unr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(unr);

    vnr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(vnr);

    wnr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(wnr); 

    Tnr = (double*)malloc(nx * (ny+2) * NZ * sizeof(double));
    Malloc_Judge(Tnr);
    
    x1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(x1);

    y11 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(y11);

    z1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(z1);

    d1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(d1); 

    u1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(u1);

    v1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(v1); 

    w1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(w1);
    
    T1 = (double*)malloc( ny1 * NZ1 * sizeof(double));
    Malloc_Judge(T1);
        
    x2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(x2);

    y2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(y2);

    z2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(z2);

    d2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(d2); 

    u2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(u2);

    v2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(v2); 

    w2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(w2);
    
    T2 = (double*)malloc( nx1 * ny1 * sizeof(double));
    Malloc_Judge(T2);

    d13d = (double*)malloc( nx1 * ny1 * NZ1 * sizeof(double));
    Malloc_Judge(d13d);

    u13d = (double*)malloc( nx1 * ny1 * NZ1 * sizeof(double));
    Malloc_Judge(u13d);

    v13d = (double*)malloc( nx1 * ny1 * NZ1 * sizeof(double));
    Malloc_Judge(v13d);

    w13d = (double*)malloc( nx1 * ny1 * NZ1 * sizeof(double));
    Malloc_Judge(w13d);

    T13d = (double*)malloc( nx1 * ny1 * NZ1 * sizeof(double));
    Malloc_Judge(T13d);
}

#undef Malloc_Judge

void Read_oldmesh(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    int KB, n1, n2, n3;
    MPI_File tmp_file;
    char filename[120];

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
    double (*xx3dr)[ny+2][nx] = (double (*)[ny+2][nx])(x3dr);
    double (*yy3dr)[ny+2][nx] = (double (*)[ny+2][nx])(y3dr);
    double (*zz3dr)[ny+2][nx] = (double (*)[ny+2][nx])(z3dr);

    double *x3d_buff, *y3d_buff, *z3d_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;


    MPI_File_open(MPI_COMM_WORLD, "Mesh3d-new-GPU.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
    if(my_id == 0) printf("READ X3d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, x3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Y3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, y3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Z3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, z3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("READ old mesh data OK \n");

//=======================================赋值================================================
        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < ny; j++){
                for(int i = 0; i < nx; i++){
                    xx3dr[k][j+1][i] = xx3d[k][j][i];
                    yy3dr[k][j+1][i] = yy3d[k][j][i];
                    zz3dr[k][j+1][i] = zz3d[k][j][i];
                }
            }
        }

        if(Iflag_half == 1){
            for(int k = 0; k < NZ; k++){
                for(int i = 0; i < nx; i++){
                    xx3dr[k][0][i] = xx3dr[k][2][i];
                    yy3dr[k][0][i] = -yy3dr[k][2][i];
                    zz3dr[k][0][i] = zz3dr[k][2][i];

                    xx3dr[k][ny+1][i] = xx3dr[k][ny-1][i];
                    yy3dr[k][ny+1][i] = -yy3dr[k][ny-1][i];
                    zz3dr[k][ny+1][i] = zz3dr[k][ny-1][i];
                }
            }
        }
        else{
            for(int k = 0; k < NZ; k++){
                for(int i = 0; i < nx; i++){
                    xx3dr[k][0][i] = xx3dr[k][ny-1][i];
                    yy3dr[k][0][i] = yy3dr[k][ny-1][i];
                    zz3dr[k][0][i] = zz3dr[k][ny-1][i];

                    xx3dr[k][ny+1][i] = xx3dr[k][2][i];
                    yy3dr[k][ny+1][i] = yy3dr[k][2][i];
                    zz3dr[k][ny+1][i] = zz3dr[k][2][i];
                }
            }

        }

    MPI_Barrier(MPI_COMM_WORLD);

//============================输出截面数据观察数据是否正确读入===============================
    
    for(int k = 0; k < NZ; k++){
        for(int i = 0; i < nx; i++){
            xx3d_buff[k][i] = xx3d[k][0][i];      //设置缓冲数组防止数据交换时覆盖掉
            yy3d_buff[k][i] = yy3d[k][0][i];
            zz3d_buff[k][i] = zz3d[k][0][i];

        }
    }
    sprintf(filename, "test-oldmesh.dat");      
    if(my_id == 0){

        printf("write test-oldmesh data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,y,z\n");
        fprintf(fp, "zone i = %d, j = %d , k = %d\n", nx, 1, nz);
        fclose(fp); 
    }
    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < nx; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", 
                            xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j]);
                        
                    }
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
        }
    

    MPI_Barrier(MPI_COMM_WORLD);
    

}

void Read_olddata(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    char filename[120];

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(dn);
    double (*u)[ny][nx] = (double (*)[ny][nx])(un);
    double (*v)[ny][nx] = (double (*)[ny][nx])(vn);
    double (*w)[ny][nx] = (double (*)[ny][nx])(wn);
    double (*T)[ny][nx] = (double (*)[ny][nx])(Tn);

    double (*xx3dr)[ny+2][nx] = (double (*)[ny+2][nx])(x3dr);
    double (*yy3dr)[ny+2][nx] = (double (*)[ny+2][nx])(y3dr);
    double (*zz3dr)[ny+2][nx] = (double (*)[ny+2][nx])(z3dr);

    double (*dr)[ny+2][nx] = (double (*)[ny+2][nx])(dnr);
    double (*ur)[ny+2][nx] = (double (*)[ny+2][nx])(unr);
    double (*vr)[ny+2][nx] = (double (*)[ny+2][nx])(vnr);
    double (*wr)[ny+2][nx] = (double (*)[ny+2][nx])(wnr);
    double (*Tr)[ny+2][nx] = (double (*)[ny+2][nx])(Tnr);
    
    double *x3d_buff, *y3d_buff, *z3d_buff;
    double *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT_buff  = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd_buff)[nx]  = (double(*)[nx])pd_buff;
    double (*ppu_buff)[nx]  = (double(*)[nx])pu_buff;
    double (*ppv_buff)[nx]  = (double(*)[nx])pv_buff;
    double (*ppw_buff)[nx]  = (double(*)[nx])pw_buff;
    double (*ppT_buff)[nx]  = (double(*)[nx])pT_buff;
    

    MPI_File_open(MPI_COMM_WORLD, "flow3d-plot3d-new-GPU.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id == 0) printf("READ d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, dn+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ u ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, un+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ v ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, vn+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ w ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, wn+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ T ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, Tn+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
    

    //============================输出截面数据观察数据是否正确读入===============================
    
    for(int k = 0; k < NZ; k++){
        for(int i = 0; i < nx; i++){
            xx3d_buff[k][i] = xx3d[k][0][i];      //设置缓冲数组防止数据交换时覆盖掉
            yy3d_buff[k][i] = yy3d[k][0][i];
            zz3d_buff[k][i] = zz3d[k][0][i];

            ppd_buff[k][i] = d[k][0][i];
            ppu_buff[k][i] = u[k][0][i];
            ppv_buff[k][i] = v[k][0][i];
            ppw_buff[k][i] = w[k][0][i];
            ppT_buff[k][i] = T[k][0][i];

        }
    }
    sprintf(filename, "test-olddata.dat");      
    if(my_id == 0){

        printf("write test-olddata data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i = %d, j = %d , k = %d\n", nx, 1, nz);
        fclose(fp); 
    }
    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < nx; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                            xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j],
                            ppv_buff[k][j], ppw_buff[k][j], ppT_buff[k][j]);
                        
                    }
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
        }

     //=======================================赋值================================================
        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < ny; j++){
                for(int i = 0; i < nx; i++){
                    dr[k][j+1][i] = d[k][j][i];
                    ur[k][j+1][i] = u[k][j][i];
                    vr[k][j+1][i] = v[k][j][i];
                    wr[k][j+1][i] = w[k][j][i];
                    Tr[k][j+1][i] = T[k][j][i];
                }
            }
        }
    //=======================================半锥==================================================
    if(Iflag_half == 1){
        for(int k = 0; k < NZ; k++){
            for(int i = 0; i < nx; i++){
                dr[k][0][i] = dr[k][2][i];
                ur[k][0][i] = ur[k][2][i];
                vr[k][0][i] = -vr[k][2][i];
                wr[k][0][i] = wr[k][2][i];
                Tr[k][0][i] = Tr[k][2][i];

                dr[k][ny+1][i] = dr[k][ny-1][i];
                ur[k][ny+1][i] = ur[k][ny-1][i];
                vr[k][ny+1][i] = -vr[k][ny-1][i];
                wr[k][ny+1][i] = wr[k][ny-1][i];
                Tr[k][ny+1][i] = Tr[k][ny-1][i];
            }
        }
    }
    //========================================全锥==================================================
    else{
        for(int k = 0; k < NZ; k++){
            for(int i = 0; i < nx; i++){
                dr[k][0][i] = dr[k][ny-1][i];
                ur[k][0][i] = ur[k][ny-1][i];
                vr[k][0][i] = vr[k][ny-1][i];
                wr[k][0][i] = wr[k][ny-1][i];
                Tr[k][0][i] = Tr[k][ny-1][i];

                dr[k][ny+1][i] = dr[k][2][i];
                ur[k][ny+1][i] = ur[k][2][i];
                vr[k][ny+1][i] = vr[k][2][i]; //原vnr[k][ny+1][i] = -vnr[k][2][i];
                wr[k][ny+1][i] = wr[k][2][i];
                Tr[k][ny+1][i] = Tr[k][2][i];
            }
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

}

/*#define FREAD(ptr , size , num , stream) \
    {   int tmp_buffer;\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
        fread(ptr , size , num , stream);\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
    }

void Read_olddata1(){               //为插值读取
    int num = nx * ny;
    
    if(my_id == 0)printf("Read olddata1-d ...\n");

}

#undef FREAD*/

void Read_newmesh(){
    double (*xx1)[ny1] = (double (*)[ny1])(x1);
    double (*yy1)[ny1] = (double (*)[ny1])(y11);
    double (*zz1)[ny1] = (double (*)[ny1])(z1);
    double (*xx2)[nx1] = (double (*)[nx1])(x2);
    double (*yy2)[nx1] = (double (*)[nx1])(y2);
    double (*zz2)[nx1] = (double (*)[nx1])(z2);

    double *x3d1, *y3d1, *z3d1;
    double *x1_buff, *y1_buff, *z1_buff;
    
    x3d1 = (double*)malloc(nx1 * ny1 * NZ1 * sizeof(double));
    y3d1 = (double*)malloc(nx1 * ny1 * NZ1 * sizeof(double));
    z3d1 = (double*)malloc(nx1 * ny1 * NZ1 * sizeof(double));
    x1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    y1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    z1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));

    double (*x3d1_buff)[ny1][nx1] = (double(*)[ny1][nx1])(x3d1);
    double (*y3d1_buff)[ny1][nx1] = (double(*)[ny1][nx1])(y3d1);
    double (*z3d1_buff)[ny1][nx1] = (double(*)[ny1][nx1])(z3d1);
    double (*xx1_buff)[ny1] = (double(*)[ny1])x1_buff;
    double (*yy1_buff)[ny1] = (double(*)[ny1])y1_buff;
    double (*zz1_buff)[ny1] = (double(*)[ny1])z1_buff;

    int num = nx1 * ny1;
    int num_byte = nx1 * ny1 * sizeof(double);
    MPI_File tmp_file;
    char filename[120];
    char filename1[120];

    int m0=nz1, n0, id0;

    MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
    if(my_id == 0) printf("READ X3d ...\n");

    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ1; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, x3d1+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Y3d ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, y3d1+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Z3d ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, z3d1+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    
    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("READ new mesh data OK \n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //==================================入口剖面========================================
    for(int k = 0; k < NZ1; k++){
        for(int j = 0; j < ny1; j++){
            xx1[k][j] = x3d1_buff[k][j][0];
            yy1[k][j] = y3d1_buff[k][j][0];
            zz1[k][j] = z3d1_buff[k][j][0];

            xx1_buff[k][j] = xx1[k][j];
            yy1_buff[k][j] = yy1[k][j];
            zz1_buff[k][j] = zz1[k][j];
        }
    }
    if(my_id == 0) printf("generate inlet OK ...\n");
    //===================================上边界=========================================
    for(int i = 0; i < n_processe; i++){
        if(NP1[i] < m0 && NPE1[i] >= m0){
            id0 = i;
        }
    }
    if(my_id == 0) printf("find id0 OK , id0 = %d\n", id0);
    
    n0 = m0 - NP1[id0] -1; //寻找所属my_id内的壁面法向编号
    if(my_id == 0) printf("generate n0 OK ,n0 = %d\n", n0);

    if(my_id == id0){
        for(int j = 0; j < ny1; j++){
            for(int i = 0; i < nx1; i++){
                xx2[j][i] = x3d1_buff[n0][j][i];
                yy2[j][i] = y3d1_buff[n0][j][i];   //需要找到(nz1-1)位于哪个my_id中，再广播给所有节点
                zz2[j][i] = z3d1_buff[n0][j][i];
            }
        }
        printf("generate xx yy zz OK ...\n");
    }
    

    //MPI_Barrier(MPI_COMM_WORLD);         //阻断
    MPI_Bcast(xx2, nx1 * ny1, MPI_DOUBLE, id0, MPI_COMM_WORLD);   //将所赋值广播给所有节点
    MPI_Bcast(yy2, nx1 * ny1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
    MPI_Bcast(zz2, nx1 * ny1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
    if(my_id == 0) printf("generate upper boundary OK ...\n");

    //==============================输出截面看文件是否正确读入 (无问题，因为写文件太花时间而注释掉)============================
    /*sprintf(filename, "newmesh-i0.dat");
    sprintf(filename1, "newmesh-k0.dat");
    if(my_id == 0){

        printf("write newmesh-i0 data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,y,z\n");
        fprintf(fp, "zone i = %d, j = %d , k = %d\n", 1, ny1, nz1);
        fclose(fp); 

        printf("write newmesh-k0 data ...\n");
        
        fp1 = fopen(filename1, "w");
       
        fprintf(fp1, "variables=x,y,z\n");
        fprintf(fp1, "zone i = %d, j = %d , k = %d\n", nx1, ny1, 1);
        for(int j = 0; j < ny1; j++){
            for(int i = 0; i < nx1; i++){
                fprintf(fp1, "%15.6f%15.6f%15.6f\n", xx2[j][i], yy2[j][i], zz2[j][i]);
            }
        }
        fclose(fp1);
    }
    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ1[n]; k++){
                    for(int j = 0; j < ny1; j++){                       
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", 
                            xx1_buff[k][j], yy1_buff[k][j], zz1_buff[k][j]);                      
                    }
                }
                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }*/

    free(x3d1);
    free(y3d1);
    free(z3d1);

    MPI_Barrier(MPI_COMM_WORLD);
}

void cz3d_to_2d_inlet(){
    double (*xx3dr)[ny+2][nx] = (double (*)[ny+2][nx])(x3dr);
    double (*yy3dr)[ny+2][nx] = (double (*)[ny+2][nx])(y3dr);
    double (*zz3dr)[ny+2][nx] = (double (*)[ny+2][nx])(z3dr);

    double (*dr)[ny+2][nx] = (double (*)[ny+2][nx])(dnr);
    double (*ur)[ny+2][nx] = (double (*)[ny+2][nx])(unr);
    double (*vr)[ny+2][nx] = (double (*)[ny+2][nx])(vnr);
    double (*wr)[ny+2][nx] = (double (*)[ny+2][nx])(wnr);
    double (*Tr)[ny+2][nx] = (double (*)[ny+2][nx])(Tnr);

    double (*xx1)[ny1] = (double (*)[ny1])(x1);
    double (*yy1)[ny1] = (double (*)[ny1])(y11);
    double (*zz1)[ny1] = (double (*)[ny1])(z1);
    
    double (*pd1)[ny1] = (double (*)[ny1])(d1);
    double (*pu1)[ny1] = (double (*)[ny1])(u1);
    double (*pv1)[ny1] = (double (*)[ny1])(v1);
    double (*pw1)[ny1] = (double (*)[ny1])(w1);
    double (*pT1)[ny1] = (double (*)[ny1])(T1);

    double *x0, *y0, *z0;
    double *d0, *u0, *v0, *w0, *T0;
    double *x0_bufftest, *y0_bufftest, *z0_bufftest;
    double *d0_bufftest, *u0_bufftest, *v0_bufftest, *w0_bufftest, *T0_bufftest;
    double *xx0_buff, *yy0_buff, *zz0_buff;
    double *dd0_buff, *uu0_buff, *vv0_buff, *ww0_buff, *TT0_buff;
    double *xx1_buff, *yy1_buff, *zz1_buff;
    double *pd1_buff, *pu1_buff, *pv1_buff, *pw1_buff, *pT1_buff;
    double *xx3dr_buff, *yy3dr_buff, *zz3dr_buff;
    double *pdr_buff, *pur_buff, *pvr_buff, *pwr_buff, *pTr_buff;

    x0 = (double*)malloc( ny * NZ * sizeof(double));
    y0 = (double*)malloc( ny * NZ * sizeof(double));
    z0 = (double*)malloc( ny * NZ * sizeof(double));
    d0 = (double*)malloc( ny * NZ * sizeof(double));
    u0 = (double*)malloc( ny * NZ * sizeof(double));
    v0 = (double*)malloc( ny * NZ * sizeof(double));
    w0 = (double*)malloc( ny * NZ * sizeof(double));
    T0 = (double*)malloc( ny * NZ * sizeof(double));

    double (*x0_buff)[ny] = (double (*)[ny])(x0);
    double (*y0_buff)[ny] = (double (*)[ny])(y0);
    double (*z0_buff)[ny] = (double (*)[ny])(z0);
    double (*d0_buff)[ny] = (double (*)[ny])(d0);
    double (*u0_buff)[ny] = (double (*)[ny])(u0);
    double (*v0_buff)[ny] = (double (*)[ny])(v0);
    double (*w0_buff)[ny] = (double (*)[ny])(w0);
    double (*T0_buff)[ny] = (double (*)[ny])(T0);

    x0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    y0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    z0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    d0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    u0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    v0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    w0_bufftest = (double*)malloc( ny * NZ * sizeof(double));
    T0_bufftest = (double*)malloc( ny * NZ * sizeof(double));

    double (*px0_bufftest)[ny] = (double (*)[ny])(x0_bufftest);
    double (*py0_bufftest)[ny] = (double (*)[ny])(y0_bufftest);
    double (*pz0_bufftest)[ny] = (double (*)[ny])(z0_bufftest);
    double (*pd0_bufftest)[ny] = (double (*)[ny])(d0_bufftest);
    double (*pu0_bufftest)[ny] = (double (*)[ny])(u0_bufftest);
    double (*pv0_bufftest)[ny] = (double (*)[ny])(v0_bufftest);
    double (*pw0_bufftest)[ny] = (double (*)[ny])(w0_bufftest);
    double (*pT0_bufftest)[ny] = (double (*)[ny])(T0_bufftest);
    
    xx0_buff = (double*)malloc( ny * NZ * sizeof(double));
    yy0_buff = (double*)malloc( ny * NZ * sizeof(double));
    zz0_buff = (double*)malloc( ny * NZ * sizeof(double));
    dd0_buff = (double*)malloc( ny * NZ * sizeof(double));
    uu0_buff = (double*)malloc( ny * NZ * sizeof(double));
    vv0_buff = (double*)malloc( ny * NZ * sizeof(double));
    ww0_buff = (double*)malloc( ny * NZ * sizeof(double));
    TT0_buff = (double*)malloc( ny * NZ * sizeof(double));

    double (*pxx0_buff)[ny] = (double (*)[ny])(xx0_buff);
    double (*pyy0_buff)[ny] = (double (*)[ny])(yy0_buff);
    double (*pzz0_buff)[ny] = (double (*)[ny])(zz0_buff);
    double (*pdd0_buff)[ny] = (double (*)[ny])(dd0_buff);
    double (*puu0_buff)[ny] = (double (*)[ny])(uu0_buff);
    double (*pvv0_buff)[ny] = (double (*)[ny])(vv0_buff);
    double (*pww0_buff)[ny] = (double (*)[ny])(ww0_buff);
    double (*pTT0_buff)[ny] = (double (*)[ny])(TT0_buff);

    xx1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    yy1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    zz1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    pd1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    pu1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    pv1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    pw1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));
    pT1_buff = (double*)malloc( ny1 * NZ1 * sizeof(double));

    double (*pxx1_buff)[ny1] = (double (*)[ny1])(xx1_buff);
    double (*pyy1_buff)[ny1] = (double (*)[ny1])(yy1_buff);
    double (*pzz1_buff)[ny1] = (double (*)[ny1])(zz1_buff);
    double (*ppd1_buff)[ny1] = (double (*)[ny1])(pd1_buff);
    double (*ppu1_buff)[ny1] = (double (*)[ny1])(pu1_buff);
    double (*ppv1_buff)[ny1] = (double (*)[ny1])(pv1_buff);
    double (*ppw1_buff)[ny1] = (double (*)[ny1])(pw1_buff);
    double (*ppT1_buff)[ny1] = (double (*)[ny1])(pT1_buff);

    xx3dr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    yy3dr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    zz3dr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    pdr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    pur_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    pvr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    pwr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));
    pTr_buff = (double*)malloc(  (ny) * NZ * sizeof(double));

    double (*pxx3dr_buff)[ny] = (double (*)[ny])(xx3dr_buff);
    double (*pyy3dr_buff)[ny] = (double (*)[ny])(yy3dr_buff);
    double (*pzz3dr_buff)[ny] = (double (*)[ny])(zz3dr_buff);
    double (*ppdr_buff)[ny] = (double (*)[ny])(pdr_buff);
    double (*ppur_buff)[ny] = (double (*)[ny])(pur_buff);
    double (*ppvr_buff)[ny] = (double (*)[ny])(pvr_buff);
    double (*ppwr_buff)[ny] = (double (*)[ny])(pwr_buff);
    double (*ppTr_buff)[ny] = (double (*)[ny])(pTr_buff);
    


    

    double a1, a2;
    int i1, i2;
    int i0;

    char filename[120];
    char filename1[120];
    char filename2[120];

    //========================= find the nearest i0 plane in the original mesh=======================
    if(my_id == 0){
        i0 = 0;
        double xd0 = fabs(xx3dr[0][1][0] - xx1[0][0]);
        for(int i = 0; i < nx; i++){
            if(fabs(xx3dr[0][1][i] - xx1[0][0]) < xd0){
                xd0 = fabs(xx3dr[0][1][i] - xx1[0][0]);
                i0 = i;
            }
        }
        if(xx1[0][0] > xx3dr[0][1][i0]){
            i1 = i0, i2 = i0 + 1;
        }
        else{
            i1 = i0 - 1, i2 = i0 ;
        }
        a1 = (xx3dr[0][1][i2] - xx1[0][0])/(xx3dr[0][1][i2] - xx3dr[0][1][i1]);
        a2 = (xx1[0][0] - xx3dr[0][1][i1])/(xx3dr[0][1][i2] - xx3dr[0][1][i1]);
        printf(" a1 = %lf, a2 = %lf, i0 = %d, i1 = %d, i2 = %d\n", a1, a2, i0, i1, i2);
    }
    //MPI_Barrier(MPI_COMM_WORLD);         //阻断
    MPI_Bcast(&a1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);   //将所赋值广播给所有节点
    MPI_Bcast(&a2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i2, 1, MPI_INT, 0, MPI_COMM_WORLD);


    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            x0_buff[k][j] = a1 * xx3dr[k][j+1][i1] + a2 * xx3dr[k][j+1][i2];
            y0_buff[k][j] = a1 * yy3dr[k][j+1][i1] + a2 * yy3dr[k][j+1][i2];
            z0_buff[k][j] = a1 * zz3dr[k][j+1][i1] + a2 * zz3dr[k][j+1][i2];
            d0_buff[k][j] = a1 * dr[k][j+1][i1] + a2 * dr[k][j+1][i2];
            u0_buff[k][j] = a1 * ur[k][j+1][i1] + a2 * ur[k][j+1][i2];
            v0_buff[k][j] = a1 * vr[k][j+1][i1] + a2 * vr[k][j+1][i2];
            w0_buff[k][j] = a1 * wr[k][j+1][i1] + a2 * wr[k][j+1][i2];
            T0_buff[k][j] = a1 * Tr[k][j+1][i1] + a2 * Tr[k][j+1][i2];

            px0_bufftest[k][j] = x0_buff[k][j];
            py0_bufftest[k][j] = y0_buff[k][j];
            pz0_bufftest[k][j] = z0_buff[k][j];
            pd0_bufftest[k][j] = d0_buff[k][j];
            pu0_bufftest[k][j] = u0_buff[k][j];
            pv0_bufftest[k][j] = v0_buff[k][j];
            pw0_bufftest[k][j] = w0_buff[k][j];
            pT0_bufftest[k][j] = T0_buff[k][j];
        }
    } 

    sprintf(filename, "test-0.dat");
    if(my_id == 0){
        printf("test-0.dat ...\n");
        
        fp = fopen(filename, "w");
        fprintf(fp, "variables=x0,y0,z0,d0,u0,v0,w0,T0\n");
        fprintf(fp, "zone i = %d, j = %d\n",  ny, nz);
        fclose(fp);
    }
    for(int n = 0; n < n_processe; n++){
            
        if(my_id == 0){

            fp = fopen(filename, "a");

            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", px0_bufftest[k][j], py0_bufftest[k][j], pz0_bufftest[k][j],
                    pd0_bufftest[k][j], pu0_bufftest[k][j], pv0_bufftest[k][j], pw0_bufftest[k][j], pT0_bufftest[k][j]);
                }
            }   
            fclose(fp);
        }
        if(my_id != 0){
            MPI_Send(x0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(d0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(u0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(v0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(w0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(T0_bufftest, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }
        if(my_id != n_processe-1){
            MPI_Recv(x0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(d0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(u0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(v0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(w0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(T0_bufftest, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }



    cz2d_to_2d(y0_buff, z0_buff, d0_buff, u0_buff, v0_buff, w0_buff, T0_buff);

    for(int j = 0; j < NZ1; j++){
        for(int i = 0; i < ny1; i++){
            if(Iflag_half == 1 && yy1[j][i] < 0){
                pv1[j][i] = -pv1[j][i];
            }

            pxx1_buff[j][i] = xx1[j][i];
            pyy1_buff[j][i] = yy1[j][i];
            pzz1_buff[j][i] = zz1[j][i];
            ppd1_buff[j][i] = pd1[j][i];
            ppu1_buff[j][i] = pu1[j][i];
            ppv1_buff[j][i] = pv1[j][i];
            ppw1_buff[j][i] = pw1[j][i];
            ppT1_buff[j][i] = pT1[j][i];

        }
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
                pxx3dr_buff[k][j] = xx3dr[k][j+1][i0];
                pyy3dr_buff[k][j] = yy3dr[k][j+1][i0];
                pzz3dr_buff[k][j] = zz3dr[k][j+1][i0];
                ppdr_buff[k][j] = dr[k][j+1][i0];
                ppur_buff[k][j] = ur[k][j+1][i0];
                ppvr_buff[k][j] = vr[k][j+1][i0];
                ppwr_buff[k][j] = wr[k][j+1][i0];
                ppTr_buff[k][j] = Tr[k][j+1][i0];

                pxx0_buff[k][j] = x0_buff[k][j];
                pyy0_buff[k][j] = y0_buff[k][j];
                pzz0_buff[k][j] = z0_buff[k][j];
                pdd0_buff[k][j] = d0_buff[k][j];
                puu0_buff[k][j] = u0_buff[k][j];
                pvv0_buff[k][j] = v0_buff[k][j];
                pww0_buff[k][j] = w0_buff[k][j];
                pTT0_buff[k][j] = T0_buff[k][j];
            
        }
    }

    if(my_id == 0){
        printf("interpolation of inlet boundary ...\n");
        printf("x0 = %lf\n", xx1[0][0]);
        printf("The nearest point is %d, x = %lf\n", i0, xx3dr[0][1][i0]);
        printf("i1 = %d, i2 = %d, a1 = %lf, a2 = %lf\n", i1, i2, a1, a2);
    }

    sprintf(filename, "flow2d-i0-new-GPU.dat");
    sprintf(filename1, "flow2d-i0-test-GPU.dat");
    sprintf(filename2, "flow2d-i0-old-GPU.dat");

    if(my_id == 0){
        printf("flow2d-i0-new-GPU.dat ...\n");
        
        fp = fopen(filename, "w");
        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i = %d, j = %d \n",  ny1, nz1);
        fclose(fp); 

        printf("flow2d-i0-test-GPU.dat ...\n");
        
        fp1 = fopen(filename1, "w");
        fprintf(fp1, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp1, "zone i = %d, j = %d \n",  ny, nz);
        fclose(fp1);

        printf("flow2d-i0-old-GPU.dat ...\n");
        
        fp2 = fopen(filename2, "w");
        fprintf(fp2, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp2, "zone i = %d, j = %d \n",  ny, nz);
        fclose(fp2);



    }
    for(int n = 0; n < n_processe; n++){
            
        if(my_id == 0){

            fp = fopen(filename, "a");

            for(int k = 0; k < NPZ1[n]; k++){
                for(int j = 0; j < ny1; j++){
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", pxx1_buff[k][j], pyy1_buff[k][j], pzz1_buff[k][j],
                    ppd1_buff[k][j], ppu1_buff[k][j], ppv1_buff[k][j], ppw1_buff[k][j], ppT1_buff[k][j]);
                }
            }   
            fclose(fp);

            fp1 = fopen(filename1, "a");

            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    fprintf(fp1, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", pxx3dr_buff[k][j], pyy3dr_buff[k][j], pzz3dr_buff[k][j],
                    ppdr_buff[k][j], ppur_buff[k][j], ppvr_buff[k][j], ppwr_buff[k][j], ppTr_buff[k][j]);
                }
            }   
            fclose(fp1);

            fp2 = fopen(filename2, "a");

            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    fprintf(fp1, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", pxx0_buff[k][j], pyy0_buff[k][j], pzz0_buff[k][j],
                    pdd0_buff[k][j], puu0_buff[k][j], pvv0_buff[k][j], pww0_buff[k][j], pTT0_buff[k][j]);
                }
            }   
            fclose(fp2);

        }    
    
        if(my_id != 0){
            MPI_Send(xx1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(yy1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(zz1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT1_buff, ny1*NPZ1[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(xx3dr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(yy3dr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(zz3dr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pdr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pur_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pvr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pwr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pTr_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(xx0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(yy0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(zz0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(dd0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(uu0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(vv0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(ww0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(TT0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            
        }

        if(my_id != n_processe-1){
            MPI_Recv(xx1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(yy1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(zz1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT1_buff, ny1*NPZ1[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(xx3dr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(yy3dr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(zz3dr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pdr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pur_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pvr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pwr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pTr_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(xx0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(yy0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(zz0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(dd0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(uu0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(vv0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(ww0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(TT0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            
        }
    }

    if(my_id == 0)printf("generate 2d inlet data OK \n");
    MPI_Barrier(MPI_COMM_WORLD);

}
 
void cz2d_to_2d(double (*Uy0)[ny], double (*Uz0)[ny], double (*Ud0)[ny], double (*Uu0)[ny], double (*Uv0)[ny], double (*Uw0)[ny], double (*UT0)[ny]){
 //函数声明可能有问题    
    double (*y0_buff)[ny] = (double (*)[ny])(Uy0);
    double (*z0_buff)[ny] = (double (*)[ny])(Uz0);
    double (*d0_buff)[ny] = (double (*)[ny])(Ud0);
    double (*u0_buff)[ny] = (double (*)[ny])(Uu0);
    double (*v0_buff)[ny] = (double (*)[ny])(Uv0);
    double (*w0_buff)[ny] = (double (*)[ny])(Uw0);
    double (*T0_buff)[ny] = (double (*)[ny])(UT0);

    double (*py_in)[ny] = (double (*)[ny])(y3d_in);
    double (*pz_in)[ny] = (double (*)[ny])(z3d_in);
    double (*pd_in)[ny] = (double (*)[ny])(d_in);
    double (*pu_in)[ny] = (double (*)[ny])(u_in);
    double (*pv_in)[ny] = (double (*)[ny])(v_in);
    double (*pw_in)[ny] = (double (*)[ny])(w_in);
    double (*pT_in)[ny] = (double (*)[ny])(T_in);
    
    double (*yy1)[ny1] = (double (*)[ny1])(y11);
    double (*zz1)[ny1] = (double (*)[ny1])(z1);
    
    double (*pd1)[ny1] = (double (*)[ny1])(d1);
    double (*pu1)[ny1] = (double (*)[ny1])(u1);
    double (*pv1)[ny1] = (double (*)[ny1])(v1);
    double (*pw1)[ny1] = (double (*)[ny1])(w1);
    double (*pT1)[ny1] = (double (*)[ny1])(T1);

    double *r0, *s0;
    double fa2[5][nz], rr2[nz];
    double r1, s1, f1[6], Uf2[5] ,Uf22[5];
    double *f2;

    char filename[120];
    //======================缓冲数组防止输出时数据交换对原数据进行覆盖===================
    double *yy0_buff, *zz0_buff, *dd0_buff, *uu0_buff, *vv0_buff, *ww0_buff, *TT0_buff;
    double *r0_buff, *s0_buff;
    double *r0_in, *s0_in;
    //double *pr0_in, *ps0_in;

    yy0_buff = (double*)malloc( ny * NZ * sizeof(double));
    zz0_buff = (double*)malloc( ny * NZ * sizeof(double));
    dd0_buff = (double*)malloc( ny * NZ * sizeof(double));
    uu0_buff = (double*)malloc( ny * NZ * sizeof(double));
    vv0_buff = (double*)malloc( ny * NZ * sizeof(double));
    ww0_buff = (double*)malloc( ny * NZ * sizeof(double));
    TT0_buff = (double*)malloc( ny * NZ * sizeof(double));

    r0 = (double*)malloc( ny * NZ * sizeof(double));
    s0 = (double*)malloc( ny * NZ * sizeof(double));

    r0_buff = (double*)malloc( ny * NZ * sizeof(double));
    s0_buff = (double*)malloc( ny * NZ * sizeof(double));


    double (*pyy0_buff)[ny] = (double (*)[ny])(yy0_buff);
    double (*pzz0_buff)[ny] = (double (*)[ny])(zz0_buff);
    double (*pdd0_buff)[ny] = (double (*)[ny])(dd0_buff);
    double (*puu0_buff)[ny] = (double (*)[ny])(uu0_buff);
    double (*pvv0_buff)[ny] = (double (*)[ny])(vv0_buff);
    double (*pww0_buff)[ny] = (double (*)[ny])(ww0_buff);
    double (*pTT0_buff)[ny] = (double (*)[ny])(TT0_buff);

    double (*pr0)[ny] = (double (*)[ny])(r0);
    double (*ps0)[ny] = (double (*)[ny])(s0);

    double (*rr0_buff)[ny] = (double (*)[ny])(r0_buff);
    double (*ss0_buff)[ny] = (double (*)[ny])(s0_buff);

    r0_in = (double*)malloc( ny * nz * sizeof(double));
    s0_in = (double*)malloc( ny * nz * sizeof(double));
    double (*pr0_in)[ny] = (double (*)[ny])(r0_in);
    double (*ps0_in)[ny] = (double (*)[ny])(s0_in);

    int NPZ_in[n_processe], NP_in[n_processe];
    

    if(my_id == 0) printf(" cz inlet ...\n");
    for(int j = 0; j < NZ; j++){
        for(int i = 0; i < ny; i++){
            pr0[j][i] = sqrt(pow(Uy0[j][i], 2) + pow(Uz0[j][i], 2));
            ps0[j][i] = acos(Uz0[j][i]/pr0[j][i]);
            if(Iflag_half == 0){
                if(Uy0[j][i] < 0){
                    ps0[j][i] = ps0[j][i] + PI;
                }
                if(i == ny-1){
                    ps0[j][i] = 2.0 * PI;
                }
            }

            pyy0_buff[j][i] = Uy0[j][i];
            pzz0_buff[j][i] = Uz0[j][i];
            rr0_buff[j][i] = pr0[j][i];
            ss0_buff[j][i] = ps0[j][i];
        }
    }

    sprintf(filename, "test-inlet-cz.dat");
    if(my_id == 0){

        printf(" out put cz test ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=y,z,seta,r\n");
        fprintf(fp, "zone i = %d, j = %d \n",  ny, nz);
        fclose(fp); 
    }
    for(int n = 0; n < n_processe; n++){
            
        if(my_id == 0){

            fp = fopen(filename, "a");

            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){                       
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", 
                        pyy0_buff[k][j], pzz0_buff[k][j], ss0_buff[k][j], rr0_buff[k][j]);                      
                }
            }
            fclose(fp);
        }    
    
        if(my_id != 0){
            MPI_Send(yy0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(zz0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(r0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(s0_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(yy0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(zz0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(r0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(s0_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    for(int i = 0; i < n_processe; i++){
        NPZ_in[i] = NPZ[i]*ny;
        NP_in[i] = NP[i]*ny;
    }

    MPI_Allgatherv( Uy0 , ny*NZ , MPI_DOUBLE , y3d_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( Uz0 , ny*NZ , MPI_DOUBLE , z3d_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( Ud0 , ny*NZ , MPI_DOUBLE , d_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( Uu0 , ny*NZ , MPI_DOUBLE , u_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( Uv0 , ny*NZ , MPI_DOUBLE , v_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( Uw0 , ny*NZ , MPI_DOUBLE , w_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( UT0 , ny*NZ , MPI_DOUBLE , T_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( pr0 , ny*NZ , MPI_DOUBLE , r0_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    MPI_Allgatherv( ps0 , ny*NZ , MPI_DOUBLE , s0_in , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);

    sprintf(filename, "test-inlet-cz.dat");
    if(my_id == 0){
        printf(" 输出ALLgatherv后的结果 ...\n");
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=y,z,d,u,v,w,t\n");
        fprintf(fp, "zone i = %d, j = %d \n",  ny, nz);
        for(int k = 0; k < nz; k++){
                for(int j = 0; j < ny; j++){                       
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                        py_in[k][j], pz_in[k][j], pd_in[k][j], pu_in[k][j], pv_in[k][j], pw_in[k][j], pT_in[k][j]);                      
                }
            }    

        fclose(fp);

    }

    if(my_id == 0) printf(" preparition for cz  OK ...\n");

    for(int j1 = 0; j1 < NZ1; j1++){
        for(int i1 = 0; i1 < ny1; i1++){
            r1 = sqrt(pow(yy1[j1][i1], 2) + pow(zz1[j1][i1], 2));
            s1 = acos(zz1[j1][i1]/r1);

            for(int j = 0; j < nz; j++){     //for(int j = 0; j < NZ; j++)
                cz2d_seta( j, s1, f1, pr0_in, ps0_in, pd_in, pu_in , pv_in, pw_in, pT_in);   //

                rr2[j] = f1[0];
                for(int m = 0; m < 5; m++ ){
                    fa2[m][j] = f1[m+1];
                }    
            }

            //if(my_id == 0) printf("  cz in wall normal  ...\n");
            f2 = inter1d_6th(5, r1, Uf2, nz, rr2, fa2);    //f2 = inter1d_6th(5, r1, Uf2, NZ, rr2, fa2);

            pd1[j1][i1] = f2[0];
            pu1[j1][i1] = f2[1];
            pv1[j1][i1] = f2[2];
            pw1[j1][i1] = f2[3];
            pT1[j1][i1] = f2[4];
        }
    }
    if(my_id == 0) printf("  cz  OK ...\n");
    //free(r0);
    //free(s0);
    //free(fa2);
    //free(rr2);
    //MPI_Barrier(MPI_COMM_WORLD);
}

double cz2d_seta(int j, double Us1, double Uf1[6], double (*Ur0)[ny], double (*Us0)[ny] ,double (*Ud0)[ny], double (*Uu0)[ny], double (*Uv0)[ny], double (*Uw0)[ny], double (*UT0)[ny]){
    int i1;
    int LAP = 3;
    double f0[6][ny+6],seta0[ny+6];

    for(int i = 0; i < ny; i++ ){
        seta0[i+3] = Us0[j][i];
        f0[0][i+3] = Ur0[j][i];
        f0[1][i+3] = Ud0[j][i];
        f0[2][i+3] = Uu0[j][i];
        f0[3][i+3] = Uv0[j][i];
        f0[4][i+3] = Uw0[j][i];
        f0[5][i+3] = UT0[j][i];
    }

    //if(my_id == 0) printf(" interpolation in seta direction ... j = %d, s1 = %lf\n",j,Us1);
    if(Iflag_half == 1){
        for(int i = 0; i < 3; i++){
            i1 = 6 - i;                             //关于i=3对称
            seta0[i] = -seta0[i1];
            f0[0][i] = f0[0][i1];  //r
            f0[1][i] = f0[1][i1];  //d
            f0[2][i] = f0[2][i1];
            f0[3][i] = -f0[3][i1];  //v
            f0[4][i] = f0[4][i1];
            f0[5][i] = f0[5][i1];
        }
        for(int i = ny+3; i < ny+6; i++){
            i1 = 2 * (ny+2) - i;                  //关于i=ny+2对称
            seta0[i] = 2.0 * PI - seta0[i1];
            f0[0][i] = f0[0][i1];  //r
            f0[1][i] = f0[1][i1];  //d
            f0[2][i] = f0[2][i1];
            f0[3][i] = -f0[3][i1];  //v
            f0[4][i] = f0[4][i1];
            f0[5][i] = f0[5][i1];
        }
    }
    else{
        for(int i = 0; i < 3; i++){
            i1 = ny - 1 + i;                      //周期条件
            seta0[i] = seta0[i1] - 2.0 * PI;
            f0[0][i] = f0[0][i1];  //r
            f0[1][i] = f0[1][i1];  //d
            f0[2][i] = f0[2][i1];
            f0[3][i] = f0[3][i1];  //v
            f0[4][i] = f0[4][i1];
            f0[5][i] = f0[5][i1];
        }
        for(int i = ny+3; i < ny+6; i++){
            i1 = i - ny + 1;                  //周期条件
            seta0[i] = seta0[i1] - 2.0 * PI;
            f0[0][i] = f0[0][i1];  //r
            f0[1][i] = f0[1][i1];  //d
            f0[2][i] = f0[2][i1];
            f0[3][i] = f0[3][i1];  //v
            f0[4][i] = f0[4][i1];
            f0[5][i] = f0[5][i1];
        }
    }


    //if(my_id == 0) printf(" Lagrange interpolation ...\n");
    Uf1 = inter1d_6th(6, Us1, Uf1, ny + 2*LAP, seta0, f0);
    //MPI_Barrier(MPI_COMM_WORLD);
}

//================================6th order Langrage interpolation in one-dimension=============================

double * inter1d_6th(int m, double UUs1, double UUf1[m], int nx, double Useta0[nx], double Uf0[m][nx]){
    
    int k, k0, ik, ikm, km, ka, kb, j;
    double Ai[6];

    if(UUs1 <= Useta0[0]){
        for(j = 0; j < m; j++){
            UUf1[j] = Uf0[j][0];
        }
        return UUf1;
    }
    //if(my_id == 0) printf(" 1\n");

    
    if(UUs1 >= Useta0[nx-1]){
        for(j = 0; j < m; j++){
            UUf1[j] = Uf0[j][nx-1];
        }
        return UUf1;
    }
    //if(my_id == 0) printf(" 2\n");

    k0 = 0;
    for(int k = 0; k < nx-1; k++){
        if(UUs1 >= Useta0[k] && UUs1 < Useta0[k+1]){    //有问题 Useta0[k+1]？？？
            k0 = k;
            goto lable;
        }
    }
    lable: //continue;
    //if(my_id == 0) printf(" k0 = %d\n", k0);

    ka = maxint(3-k0, 1);           //可能有问题
    kb = minint(nx+2-k0, 6);
    for(j = 0; j < m; j++){
        UUf1[j] = 0.0;
        Ai[j] = 0.0;
    }
    //if(my_id == 0) printf(" ka = %d, kb = %d\n", ka, kb);

    for(k = ka; k < kb+1; k++){
        //if(my_id == 0) printf("1 k = %d, k0 = %d\n", k, k0);
        ik = k0 + k -3;                     //可能有问题
        //if(my_id == 0) printf("2 k = %d, k0 = %d, ik = %d\n", k, k0, ik);
        Ai[k - ka] = 1.0;
        //if(my_id == 0) printf("3 k = %d, k0 = %d, ik = %d\n", k, k0, ik);
        
        for(km = ka; km < kb+1; km++){
            ikm = k0 + km -3;
            //if(my_id == 0) printf(" km = %d,ikm = %d\n", km ,ikm);
            if(km != k){
                Ai[k - ka] = Ai[k - ka] * (UUs1 - Useta0[ikm])/(Useta0[ik] - Useta0[ikm]);
            //if(my_id == 0) printf(" Ai = %lf\n", Ai[k]);
            }
        }
        //if(my_id == 0) printf(" 6\n");
        for(j = 0; j < m; j++){
            UUf1[j] = UUf1[j] + Ai[k - ka] * Uf0[j][ik];
            //if(my_id == 0) printf(" j = %d, UUf1 = %lf\n",j,UUf1[0]);
        }
        //if(my_id == 0) printf(" 7\n");
    }
    
    return UUf1;

}

int maxint(int a, int b){
    if(a > b){
        return a;
    }
    else{
        return b;
    }
}

int minint(int a, int b){
    if(a < b){
        return a;
    }
    else{
        return b;
    }
}

void write_inletsection(){
    int num =  ny1;
    int num_byte =  ny1 * sizeof(double);
    
    MPI_File tmp_file;
   

    double (*xx1)[ny1] = (double (*)[ny1])(x1);
    double (*yy1)[ny1] = (double (*)[ny1])(y11);
    double (*zz1)[ny1] = (double (*)[ny1])(z1);
    
    double (*pd1)[ny1] = (double (*)[ny1])(d1);
    double (*pu1)[ny1] = (double (*)[ny1])(u1);
    double (*pv1)[ny1] = (double (*)[ny1])(v1);
    double (*pw1)[ny1] = (double (*)[ny1])(w1);
    double (*pT1)[ny1] = (double (*)[ny1])(T1);



    MPI_File_open(MPI_COMM_WORLD, "flow-inlet-section.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
    if(my_id == 0) printf("WRITE D2d ...\n");

    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
    if(my_id == 0) printf("1\n");
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, d1+num*k, num,  MPI_DOUBLE, &status);       
    }
    if(my_id == 0) printf("2\n");
    MPI_File_seek(tmp_file, NP1[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE U2d ...\n");

    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, u1+num*k, num,  MPI_DOUBLE, &status);  
    }
    MPI_File_seek(tmp_file, NP1[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE V2d ...\n");

    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, v1+num*k, num,  MPI_DOUBLE, &status);  
    }
    MPI_File_seek(tmp_file, NP1[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE W2d ...\n");

    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, w1+num*k, num,  MPI_DOUBLE, &status);  
    }
    MPI_File_seek(tmp_file, NP1[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE T2d ...\n");

    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, T1+num*k, num,  MPI_DOUBLE, &status);  
    }
    MPI_File_seek(tmp_file, NP1[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
    MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);


    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("WRITE flow inlet section OK \n");

    MPI_Barrier(MPI_COMM_WORLD);
}

void cz3d_to_2d(){
    double (*xx3dr)[ny+2][nx] = (double (*)[ny+2][nx])(x3dr);
    double (*yy3dr)[ny+2][nx] = (double (*)[ny+2][nx])(y3dr);
    double (*zz3dr)[ny+2][nx] = (double (*)[ny+2][nx])(z3dr);

    double (*dr)[ny+2][nx] = (double (*)[ny+2][nx])(dnr);
    double (*ur)[ny+2][nx] = (double (*)[ny+2][nx])(unr);
    double (*vr)[ny+2][nx] = (double (*)[ny+2][nx])(vnr);
    double (*wr)[ny+2][nx] = (double (*)[ny+2][nx])(wnr);
    double (*Tr)[ny+2][nx] = (double (*)[ny+2][nx])(Tnr);

    double (*xx2)[nx1] = (double (*)[nx1])(x2);
    double (*yy2)[nx1] = (double (*)[nx1])(y2);
    double (*zz2)[nx1] = (double (*)[nx1])(z2);
    
    double (*pd2)[nx1] = (double (*)[nx1])(d2);
    double (*pu2)[nx1] = (double (*)[nx1])(u2);
    double (*pv2)[nx1] = (double (*)[nx1])(v2);
    double (*pw2)[nx1] = (double (*)[nx1])(w2);
    double (*pT2)[nx1] = (double (*)[nx1])(T2);

    int *ist, *jst, *kst;

    ist = (int*)malloc( nx1 * ny1 * sizeof(int));
    jst = (int*)malloc( nx1 * ny1 * sizeof(int));
    kst = (int*)malloc( nx1 * ny1 * sizeof(int));

    int (*pist)[nx1] = (int (*)[nx1])(ist);
    int (*pjst)[nx1] = (int (*)[nx1])(jst);
    int (*pkst)[nx1] = (int (*)[nx1])(kst);

    int i0, j0, i, j, k, i1, j1, k1, i2, j2, k2;
    double xi, xj, xk, yi, yj, yk, zi, zj, zk, jac, ix, iy, iz, jx, jy, jz, kx, ky, kz;
    double di, ui, vi, wi, Ti, dj, uj, vj, wj, Tj, dk, uk, vk, wk, Tk;
    double dx, dy, dz, ux, uy, uz, vx, vy, vz, wx, wy, wz, Tx, Ty, Tz;
    double hx, hy, hz, deltx, delty, deltz;

    int id0, n0, id1, n1, id2, n2;
    double xx3drk1, yy3drk1, zz3drk1, drk1, urk1, vrk1, wrk1, Trk1;
    double xx3drk2, yy3drk2, zz3drk2, drk2, urk2, vrk2, wrk2, Trk2;
    double xx3drk, yy3drk, zz3drk, drk, urk, vrk, wrk, Trk;

    char filename[120];

    hy = 0.5;

    find_nearest_ijk(pist,pjst,pkst);

    for(j0 = 0; j0 < ny1; j0++){
        for(i0 = 0; i0 < nx1; i0++){
            i = pist[j0][i0];
            j = pjst[j0][i0];  
            k = pkst[j0][i0];
            

            i1 = i-1;
            j1 = j-1;
            k1 = k-1;
            

            i2 = i+1;
            j2 = j+1;
            k2 = k+1;
            

            if(i == 0) i1 = 0;
            if(i == nx-1) i2 = nx-1;
            if(k == 0) k1 = 0;
            if(k == nz-1) k2 = nz-1;

            for(int iid = 0; iid < n_processe; iid++){
                if(NP[iid] <= k && NPE[iid] > k){
                    id0 = iid;    //k所属的id号
                }
                //else id0 = 0;
            }
            n0 = k - NP[id0] ;  //k所属的id号上的壁面法向编号

            for(int iid = 0; iid < n_processe; iid++){
                if(NP[iid] <= k1 && NPE[iid] > k1){
                    id1 = iid;
                }
                //else id1 = 0;
            }
            n1 = k1 - NP[id1] ;

            for(int iid = 0; iid < n_processe; iid++){
                if(NP[iid] <= k2 && NPE[iid] > k2){
                    id2 = iid;
                }
                //else id2 = 0;
            }
            n2 = k2 - NP[id2] ;



            if(i == 0 || i == nx-1){
                hx = 1.0;
            }
            else{
                hx = 0.5;
            }

            if(k == 0 || k == nz-1){
                hz = 1.0;
            }
            else{
                hz = 0.5;
            }

            /*if(my_id == 0) {
                printf("i0 = %d, j0 = %d \n", i0, j0);
                printf("i = %d, i1 = %d, i2 = %d \n", i, i1, i2);
                printf("j = %d, j1 = %d, j2 = %d \n", j, j1, j2);
                printf("k = %d, k1 = %d, k2 = %d \n", k, k1, k2);
                printf("id0 = %d, id1 = %d, id2 = %d \n", id0, id1, id2);
                printf("n0 = %d, n1 = %d, n2 = %d \n", n0, n1, n2);
            }*/

            if(my_id == id0){

                xi = (xx3dr[n0][j][i2] - xx3dr[n0][j][i1]) * hx;
                yi = (yy3dr[n0][j][i2] - yy3dr[n0][j][i1]) * hx;
                zi = (zz3dr[n0][j][i2] - zz3dr[n0][j][i1]) * hx;
                di = (dr[n0][j][i2] - dr[n0][j][i1]) * hx;
                ui = (ur[n0][j][i2] - ur[n0][j][i1]) * hx;
                vi = (vr[n0][j][i2] - vr[n0][j][i1]) * hx;
                wi = (wr[n0][j][i2] - wr[n0][j][i1]) * hx;
                Ti = (Tr[n0][j][i2] - Tr[n0][j][i1]) * hx;

                xj = (xx3dr[n0][j2][i] - xx3dr[n0][j1][i]) * hy;
                yj = (yy3dr[n0][j2][i] - yy3dr[n0][j1][i]) * hy;
                zj = (zz3dr[n0][j2][i] - zz3dr[n0][j1][i]) * hy;
                dj = (dr[n0][j2][i] - dr[n0][j1][i]) * hy;
                uj = (ur[n0][j2][i] - ur[n0][j1][i]) * hy;
                vj = (vr[n0][j2][i] - vr[n0][j1][i]) * hy;
                wj = (wr[n0][j2][i] - wr[n0][j1][i]) * hy;
                Tj = (Tr[n0][j2][i] - Tr[n0][j1][i]) * hy;

                xx3drk = xx3dr[n0][j][i];
                yy3drk = yy3dr[n0][j][i];
                zz3drk = zz3dr[n0][j][i];
                drk = dr[n0][j][i];
                urk = ur[n0][j][i];
                vrk = vr[n0][j][i];
                wrk = wr[n0][j][i];
                Trk = Tr[n0][j][i];
            }
                //MPI_Barrier(MPI_COMM_WORLD);         //阻断
                MPI_Bcast(&xi, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(&yi, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&zi, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&di, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&ui, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&vi, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&wi, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&Ti, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&xj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);   
                MPI_Bcast(&yj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&zj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&dj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&uj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&vj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&wj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&Tj, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&xx3drk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);   
                MPI_Bcast(&yy3drk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&zz3drk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&drk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&urk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&vrk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&wrk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);
                MPI_Bcast(&Trk, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);

            //if(my_id == 0) printf("id0 = %d OK \n", id0);

            if(my_id == id1){
                xx3drk1 = xx3dr[n1][j][i];
                yy3drk1 = yy3dr[n1][j][i];
                zz3drk1 = zz3dr[n1][j][i];
                drk1 = dr[n1][j][i];
                urk1 = ur[n1][j][i];
                vrk1 = vr[n1][j][i];
                wrk1 = wr[n1][j][i];
                Trk1 = Tr[n1][j][i];
            }

                //MPI_Barrier(MPI_COMM_WORLD);         //阻断
                MPI_Bcast(&xx3drk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(&yy3drk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&zz3drk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&drk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&urk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&vrk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&wrk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(&Trk1, 1, MPI_DOUBLE, id1, MPI_COMM_WORLD);
            
            //if(my_id == 0) printf("id1 = %d OK \n", id1);

            if(my_id == id2){
                xx3drk2 = xx3dr[n2][j][i];
                yy3drk2 = yy3dr[n2][j][i];
                zz3drk2 = zz3dr[n2][j][i];
                drk2 = dr[n2][j][i];
                urk2 = ur[n2][j][i];
                vrk2 = vr[n2][j][i];
                wrk2 = wr[n2][j][i];
                Trk2 = Tr[n2][j][i];
            }

                //MPI_Barrier(MPI_COMM_WORLD);         //阻断
                MPI_Bcast(&xx3drk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(&yy3drk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&zz3drk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&drk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&urk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&vrk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&wrk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(&Trk2, 1, MPI_DOUBLE, id2, MPI_COMM_WORLD);

            //if(my_id == 0) printf("id2 = %d OK \n", id2);

            xk = (xx3drk2 - xx3drk1) * hz;
            yk = (yy3drk2 - yy3drk1) * hz;
            zk = (zz3drk2 - zz3drk1) * hz;
            dk = (drk2 - drk1) * hz;
            uk = (urk2 - urk1) * hz;
            vk = (vrk2 - vrk1) * hz;
            wk = (wrk2 - wrk1) * hz;
            Tk = (Trk2 - Trk1) * hz;

            /*xi = (xx3dr[k][j][i2] - xx3dr[k][j][i1]) * hx;
            yi = (yy3dr[k][j][i2] - yy3dr[k][j][i1]) * hx;
            zi = (zz3dr[k][j][i2] - zz3dr[k][j][i1]) * hx;
            di = (dr[k][j][i2] - dr[k][j][i1]) * hx;
            ui = (ur[k][j][i2] - ur[k][j][i1]) * hx;
            vi = (vr[k][j][i2] - vr[k][j][i1]) * hx;
            wi = (wr[k][j][i2] - wr[k][j][i1]) * hx;
            Ti = (Tr[k][j][i2] - Tr[k][j][i1]) * hx;

            xj = (xx3dr[k][j2][i] - xx3dr[k][j1][i]) * hy;
            yj = (yy3dr[k][j2][i] - yy3dr[k][j1][i]) * hy;
            zj = (zz3dr[k][j2][i] - zz3dr[k][j1][i]) * hy;
            dj = (dr[k][j2][i] - dr[k][j1][i]) * hy;
            uj = (ur[k][j2][i] - ur[k][j1][i]) * hy;
            vj = (vr[k][j2][i] - vr[k][j1][i]) * hy;
            wj = (wr[k][j2][i] - wr[k][j1][i]) * hy;
            Tj = (Tr[k][j2][i] - Tr[k][j1][i]) * hy;
            
            xk = (xx3dr[k2][j][i] - xx3dr[k1][j][i]) * hz;
            yk = (yy3dr[k2][j][i] - yy3dr[k1][j][i]) * hz;
            zk = (zz3dr[k2][j][i] - zz3dr[k1][j][i]) * hz;
            dk = (dr[k2][j][i] - dr[k1][j][i]) * hz;
            uk = (ur[k2][j][i] - ur[k1][j][i]) * hz;
            vk = (vr[k2][j][i] - vr[k1][j][i]) * hz;
            wk = (wr[k2][j][i] - wr[k1][j][i]) * hz;
            Tk = (Tr[k2][j][i] - Tr[k1][j][i]) * hz;*/

            jac = 1.0 / (xi*yj*zk + yi*zj*xk + zi*xj*yk - zi*yj*xk - yi*xj*zk - xi*zj*yk);

            ix = jac * (yj*zk - zj*yk);
            iy = jac * (zj*xk - xj*zk);
	        iz = jac * (xj*yk - yj*xk);
	        jx = jac * (yk*zi - zk*yi);
	        jy = jac * (zk*xi - xk*zi);
	        jz = jac * (xk*yi - yk*xi);
	        kx = jac * (yi*zj - zi*yj);
	        ky = jac * (zi*xj - xi*zj);
	        kz = jac * (xi*yj - yi*xj);

            dx = di*ix + dj*jx + dk*kx;
            ux = ui*ix + uj*jx + uk*kx;
            vx = vi*ix + vj*jx + vk*kx;
            wx = wi*ix + wj*jx + wk*kx;
            Tx = Ti*ix + Tj*jx + Tk*kx;

            dy = di*iy + dj*jy + dk*ky;
            uy = ui*iy + uj*jy + uk*ky;
            vy = vi*iy + vj*jy + vk*ky;
            wy = wi*iy + wj*jy + wk*ky;
            Ty = Ti*iy + Tj*jy + Tk*ky;

            dz = di*iz + dj*jz + dk*kz;
            uz = ui*iz + uj*jz + uk*kz;
            vz = vi*iz + vj*jz + vk*kz;
            wz = wi*iz + wj*jz + wk*kz;
            Tz = Ti*iz + Tj*jz + Tk*kz;

            deltx = xx2[j0][i0] - xx3drk;
            deltz = zz2[j0][i0] - zz3drk;

            if(Iflag_half == 0){
                delty = yy2[j0][i0] - yy3drk;
            }
            else{
                delty = fabs(yy2[j0][i0]) - yy3drk;
            }

            pd2[j0][i0] = drk + dx*deltx + dy*delty + dz*deltz; 
            pu2[j0][i0] = urk + ux*deltx + uy*delty + uz*deltz;
            pv2[j0][i0] = vrk + vx*deltx + vy*delty + vz*deltz;
            pw2[j0][i0] = wrk + wx*deltx + wy*delty + wz*deltz;
            pT2[j0][i0] = Trk + Tx*deltx + Ty*delty + Tz*deltz;

            /*deltx = xx2[j0][i0] - xx3dr[k][j][i];
            deltz = zz2[j0][i0] - zz3dr[k][j][i];        
            
            if(Iflag_half == 0){
                delty = yy2[j0][i0] - yy3dr[k][j][i];
            }
            else{
                delty = fabs(yy2[j0][i0] - yy3dr[k][j][i]);
            }
            
            pd2[j0][i0] = dr[k][j][i] + dx*deltx + dy*delty + dz*deltz; 
            pu2[j0][i0] = ur[k][j][i] + ux*deltx + uy*delty + uz*deltz;
            pv2[j0][i0] = vr[k][j][i] + vx*deltx + vy*delty + vz*deltz;
            pw2[j0][i0] = wr[k][j][i] + wx*deltx + wy*delty + wz*deltz;
            pT2[j0][i0] = Tr[k][j][i] + Tx*deltx + Ty*delty + Tz*deltz;*/

            if(Iflag_half == 1 && yy2[j0][i0] < 0) pv2[j0][i0] = -pv2[j0][i0];
            
        }
    }

    free(ist);
    free(jst);
    free(kst);

    if(my_id == 0) printf("Interpolation 3D OK \n");

    MPI_Barrier(MPI_COMM_WORLD);

}

void output_outflow(){

    double (*xx2)[nx1] = (double (*)[nx1])(x2);
    double (*yy2)[nx1] = (double (*)[nx1])(y2);
    double (*zz2)[nx1] = (double (*)[nx1])(z2);
    
    double (*pd2)[nx1] = (double (*)[nx1])(d2);
    double (*pu2)[nx1] = (double (*)[nx1])(u2);
    double (*pv2)[nx1] = (double (*)[nx1])(v2);
    double (*pw2)[nx1] = (double (*)[nx1])(w2);
    double (*pT2)[nx1] = (double (*)[nx1])(T2);

    char filename[120];

    sprintf(filename, "flow2d-out-new.dat");
    if(my_id == 0){

        printf(" out flow2d-out-new.dat ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i = %d, j = %d \n",  nx1, ny1);
        for(int j = 0; j < ny1; j++){
            for(int i = 0; i < nx1; i++){
                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx2[j][i], yy2[j][i], zz2[j][i],
                    pd2[j][i], pu2[j][i], pv2[j][i], pw2[j][i], pT2[j][i]);
            }
        }
        fclose(fp);

        printf(" output flow2d-out-new.dat OK\n"); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

//==================================== Find the nearest point ======================================== 

void find_nearest_ijk(int (*Uist)[nx1], int (*Ujst)[nx1], int (*Ukst)[nx1]){
    double (*xx2)[nx1] = (double (*)[nx1])(x2);
    double (*yy2)[nx1] = (double (*)[nx1])(y2);
    double (*zz2)[nx1] = (double (*)[nx1])(z2);

    double (*xx3dr)[ny+2][nx] = (double (*)[ny+2][nx])(x3dr);
    double (*yy3dr)[ny+2][nx] = (double (*)[ny+2][nx])(y3dr);
    double (*zz3dr)[ny+2][nx] = (double (*)[ny+2][nx])(z3dr);

    double d0, dd;
    int LP = 10;
    int i, j, k, i1, j1, k1, ia, ib, ja, jb, ka, kb, i0, j0, k0;
    int m0, n0, id0;

    if(my_id == 0) printf("find nearest ijk ... \n");

    for(j1 = 0; j1 < ny1; j1++){
        if(my_id == 0 && ((j1%100) == 0) ) printf(" j1 = %d  \n",  j1);
        for(i1 = 0; i1 < nx1; i1++){
            
            if(i1 != 0){
                ia = maxint( Uist[j1][i1-1]-LP, 0 );       //可能有问题
                ja = maxint( Ujst[j1][i1-1]-LP, 1 );        //做了修改
                ka = maxint( Ukst[j1][i1-1]-LP, 0 );

                ib = minint( Uist[j1][i1-1]+LP, nx-1 );
                jb = minint( Ujst[j1][i1-1]+LP, ny );       //做了修改
                kb = minint( Ukst[j1][i1-1]+LP, nz-1 );
            }
            else if(j1 != 0){
                ia = maxint( Uist[j1-1][i1]-LP, 0 );       //可能有问题
                ja = maxint( Ujst[j1-1][i1]-LP, 1 );        //做了修改
                ka = maxint( Ukst[j1-1][i1]-LP, 0 );

                ib = minint( Uist[j1-1][i1]+LP, nx-1 );
                jb = minint( Ujst[j1-1][i1]+LP, ny );       //做了修改
                kb = minint( Ukst[j1-1][i1]+LP, nz-1 );
            }
            else{
                ia = 0;
                ja = 1;   //做了修改
                ka = 0;

                ib = nx-1;
                jb = ny;  //做了修改
                kb = nz-1;
            }

            d0 = 1.0 * pow(10,20);    //d0为一个非常大的数

            //if(my_id == 0) printf("ia = %d, ib = %d, ja = %d, jb = %d, ka = %d, kb = %d, d0 = %lf \n", ia, ib, ja, jb, ka, kb, d0);
            //if(my_id == 0) printf("n_processe = %d, NP[0] = %d, NP[1] = %d, NP[2] = %d, NP[3] = %d, NP[4] = %d\n", n_processe, NP[0], NP[1], NP[2], NP[3], NP[4]);
            //if(my_id == 0) printf("NPE[0] = %d, NPE[1] = %d, NPE[2] = %d, NPE[3] = %d, NPE[4] = %d\n", NPE[0], NPE[1], NPE[2], NPE[3], NPE[4]);

            for(k = ka; k < kb+1; k++){          //需要判断k位于哪个节点上，要改！
            //if(my_id == 0) printf("k = %d\n",k);
                for(int id = 0; id < n_processe; id++){
                    if(NP[id] <= k && NPE[id] > k){
                       id0 = id; 
                    }
                    //else id0 = 0;    
                }
            //if(my_id == 0) printf("id0 = %d\n",id0);
                n0 = k - NP[id0] ; //寻找所属my_id内的壁面法向编
            //if(my_id == 0) printf("id0 = %d, n0 = %d\n",id0, n0);
                if(my_id == id0){

                for(j = ja; j < jb+1; j++){
                    for(i = ia; i < ib+1; i++){
                        //printf("i = %d, j = %d\n",i, j);
                        if(Iflag_half == 0){
                            dd = pow((xx2[j1][i1] - xx3dr[n0][j][i]), 2) + pow((yy2[j1][i1] - yy3dr[n0][j][i]), 2) + pow((zz2[j1][i1] - zz3dr[n0][j][i]), 2);
                        }
                        else{
                            dd = pow((xx2[j1][i1] - xx3dr[n0][j][i]), 2) + pow((fabs(yy2[j1][i1]) - yy3dr[n0][j][i]), 2) + pow((zz2[j1][i1] - zz3dr[n0][j][i]), 2);
                        }

                        if(dd < d0){
                            d0 = dd;
                            i0 = i;
                            j0 = j;
                            k0 = k;
                        }
                        //printf("d0 = %d, i0 = %d, j0 = %d, k0 = %d\n",d0, i0, j0, k0);

                    }
                }
            
                //MPI_Barrier(MPI_COMM_WORLD);         //阻断
                }
            //if(my_id == 0) printf("1\n");
                MPI_Bcast(&d0, 1, MPI_DOUBLE, id0, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(&i0, 1, MPI_INT, id0, MPI_COMM_WORLD);
                MPI_Bcast(&j0, 1, MPI_INT, id0, MPI_COMM_WORLD);
                MPI_Bcast(&k0, 1, MPI_INT, id0, MPI_COMM_WORLD);
            //if(my_id == 0) printf("2\n");
            }
            //if(my_id == 0) printf("22\n");
            /*for(k = ka; k < kb+1; k++){
                for(j = ja; j < jb+1; j++){
                    for(i = ia; i < ib+1; i++){
                        if(Iflag_half == 0){
                            dd = pow((xx2[j1][i1] - xx3dr[k][j][i]), 2) + pow((yy2[j1][i1] - yy3dr[k][j][i]), 2) + pow((zz2[j1][i1] - zz3dr[k][j][i]), 2);
                        }
                        else{
                            dd = pow((xx2[j1][i1] - xx3dr[k][j][i]), 2) + pow((fabs(yy2[j1][i1]) - yy3dr[k][j][i]), 2) + pow((zz2[j1][i1] - zz3dr[k][j][i]), 2);
                        }

                        if(dd < d0){
                            d0 = dd;
                            i0 = i;
                            j0 = j;
                            k0 = k;
                        }

                    }
                }

            }*/

            Uist[j1][i1] = i0;
            Ujst[j1][i1] = j0;
            Ukst[j1][i1] = k0;
            //if(my_id == 0) printf("33\n");
        }
    }

    if(my_id == 0) printf("find ijk OK \n");
    MPI_Barrier(MPI_COMM_WORLD);
}

#define FWRITE(ptr , size , num , stream) \
    {   int tmp_buffer;\
        fwrite(&tmp_buffer , sizeof(int) , 1 , stream);\
        fwrite(ptr , size , num , stream);\
        fwrite(&tmp_buffer , sizeof(int) , 1 , stream);\
    }

void write_outboundary(){
    int num =  nx1 * ny1;
    int num_byte =  nx1 * sizeof(double);
   

    double (*xx2)[nx1] = (double (*)[nx1])(x2);
    double (*yy2)[nx1] = (double (*)[nx1])(y2);
    double (*zz2)[nx1] = (double (*)[nx1])(z2);
    
    double (*pd2)[nx1] = (double (*)[nx1])(d2);
    double (*pu2)[nx1] = (double (*)[nx1])(u2);
    double (*pv2)[nx1] = (double (*)[nx1])(v2);
    double (*pw2)[nx1] = (double (*)[nx1])(w2);
    double (*pT2)[nx1] = (double (*)[nx1])(T2);

    if(my_id == 0){
        fp = fopen("flow-outboundary.dat", "w");

        printf("write flow-outboundary.dat-d ...\n");
       
            FWRITE(d2  , sizeof(double), num, fp);   

        printf("write flow-outboundary.dat-u ...\n");
        
            FWRITE(u2  , sizeof(double), num, fp);        

        printf("write flow-outboundary.dat-v ...\n");
        
            FWRITE(v2  , sizeof(double), num, fp);    

        printf("write flow-outboundary.dat-w ...\n");
        
            FWRITE(w2  , sizeof(double), num, fp);

        printf("write flow-outboundary.dat-T ...\n");
        
            FWRITE(T2  , sizeof(double), num, fp);    

        fclose(fp);
    }

    if(my_id == 0) printf("WRITE flow outer boundary OK \n");
    MPI_Barrier(MPI_COMM_WORLD);

}
#undef FWRITE

void write_3d_init_file(){
    int num = nx1 * ny1;
    int num_byte = nx1 * ny1 * sizeof(double);
    MPI_File tmp_file;

    double (*p3dd1)[ny1][nx1] = (double (*)[ny1][nx1])(d13d);
    double (*p3du1)[ny1][nx1] = (double (*)[ny1][nx1])(u13d);
    double (*p3dv1)[ny1][nx1] = (double (*)[ny1][nx1])(v13d);
    double (*p3dw1)[ny1][nx1] = (double (*)[ny1][nx1])(w13d);
    double (*p3dT1)[ny1][nx1] = (double (*)[ny1][nx1])(T13d);

    double (*pd1)[ny1] = (double (*)[ny1])(d1);
    double (*pu1)[ny1] = (double (*)[ny1])(u1);
    double (*pv1)[ny1] = (double (*)[ny1])(v1);
    double (*pw1)[ny1] = (double (*)[ny1])(w1);
    double (*pT1)[ny1] = (double (*)[ny1])(T1);

    for(int k = 0; k < NZ1; k++){
        for(int j = 0; j < ny1; j++){
            for(int i = 0; i < nx1; i++){
                p3dd1[k][j][i] = pd1[k][j];
                p3du1[k][j][i] = pu1[k][j];
                p3dv1[k][j][i] = pv1[k][j];
                p3dw1[k][j][i] = pw1[k][j];
                p3dT1[k][j][i] = pT1[k][j];
            }
        }
    }


    MPI_File_open(MPI_COMM_WORLD, "flow3d0.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_write_all(tmp_file, init, 1, MPI_INT, &status);           //阻塞通信
    MPI_File_write_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE 3d-d ...\n");

    MPI_File_seek(tmp_file, NP1[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, d13d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE 3d-u ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, u13d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE 3d-v ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, v13d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE 3d-w ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, w13d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz1-NZ1)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE 3d-T ...\n");

    for(int k = 0; k < NZ1; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, T13d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);

    if(my_id == 0) printf("WRITE init 3d flow OK \n");
    MPI_Barrier(MPI_COMM_WORLD);
}


void Finalize(){
    free(x3d);
    free(y3d);
    free(z3d);

    free(x3dr);
    free(y3dr);
    free(z3dr);
    free(head);
    free(dn);
    free(un);
    free(vn);
    free(wn);
    free(Tn);
    free(dnr);
    free(unr);
    free(vnr);
    free(wnr);
    free(Tnr);

    free(x1);
    free(y11);
    free(z1);
    
    free(d1);
    free(u1);
    free(v1);
    free(w1);
    free(T1);

    free(x2);
    free(y2);
    free(z2);
    
    free(d2);
    free(u2);
    free(v2);
    free(w2);
    free(T2);

    free(d13d);
    free(u13d);
    free(v13d);
    free(w13d);
    free(T13d);

    /*free(x0);
    free(y0);
    free(z0);
    free(d0);
    free(u0);
    free(v0);
    free(w0);
    free(T0);
    free(xx0_buff);
    free(yy0_buff);
    free(zz0_buff);
    free(dd0_buff);
    free(uu0_buff);
    free(vv0_buff);
    free(ww0_buff);
    free(TT0_buff);*/
}
