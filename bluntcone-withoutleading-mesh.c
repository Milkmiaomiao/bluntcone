//Mesh for bluntcone
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pthread.h"

#define PI 3.1415926535897932

FILE *fp;
MPI_File tmp_file;
MPI_Status status;

char str[1000];

int my_id, n_processe;

int nx, ny, nz, N, NZ, *NPZ, *NP;

int Iflag_mesh_seta, nx_buff, ngrid_buffer;

double seta_cone,seta_leeward,seta_windward,h_leeward,h_windward,hwall;
double R0, x_inlet, x_end, alfax_buff;
double seta_comput_domain;


double *x3d, *y3d, *z3d; 

void mpi_init(int *Argc, char ***Argv);
void Read_parameter();
void Data_malloc();
void get_seta_mesh( double Useta[]);
void get_seta_new( double yseta1[]);
void getsx(double dsx[]);
void generate_mesh(double Useta1[],double Udsx[]);
double Rh_hermit( double Rh, double x );
void getsy(double sy[], double SL, double deltx);
void output_grid_i0(int i0);
void output_grid_j0(int j0);
void output_mesh3d();
void Finalize();

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();
    double sx[nx];
    double seta1d[ny];

    Data_malloc();

    get_seta_mesh( seta1d );

    getsx( sx );

    generate_mesh(seta1d,sx);

    output_grid_i0(1);

    output_grid_j0(1);

    output_mesh3d();

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
        if((fp = fopen("grid2d.in", "r")) == NULL){
            printf("Can't open this file: 'grid2d.in'\n");
            exit(0);
        }
    
        fgets(str, 1000, fp);
        fgets(str, 1000, fp);
        fscanf(fp, "%d%d%d\n", &nx,&ny,&nz);
        printf("Computation start...\nnx is %d\nny is %d\nnz is %d\n",nx,ny,nz);
    
        fgets(str, 1000, fp);
        fscanf(fp, "%lf%lf%lf%lf%lf%lf\n", &seta_cone, &seta_leeward, &seta_windward, &h_leeward, &h_windward, &hwall);
        printf("seta_cone is %lf\nseta_leeward is %lf\nseta_windward is %lf\nh_leeward is %lf\nh_windward is %lf\nhwall is %lf\n",
        seta_cone,seta_leeward,seta_windward,h_leeward,h_windward,hwall);
    
        fgets(str, 1000, fp);
        fscanf(fp, "%lf%lf%lf%d%lf\n", &R0, &x_inlet, &x_end, &nx_buff ,&alfax_buff);
        printf("R0 is %lf\nx_inlet is %lf\nx_end is %lf\nnx_buf is %d\nalfax_buff is %lf\n",
        R0,x_inlet,x_end,nx_buff,alfax_buff);
    
        fgets(str, 1000, fp);
        fscanf(fp, "%d%d%lf\n", &Iflag_mesh_seta, &ngrid_buffer ,&seta_comput_domain);
        printf("Iflag_mesh_seta is %d\nngrid_buffer is %d\nseta_comput_domain is %lf\n",Iflag_mesh_seta,ngrid_buffer,seta_comput_domain);
    
            
        fclose(fp);

        printf("Read_parameter is OK!\n");

    }

    int tmp1[7];
    double tmp2[11];
    
    if(my_id == 0){
        tmp1[0] = nx;
        tmp1[1] = ny;
        tmp1[2] = nz;

        tmp1[3] = N;

        tmp1[4] = nx_buff;
        tmp1[5] = Iflag_mesh_seta;
        tmp1[6] = ngrid_buffer;


        tmp2[0] = seta_cone;
        tmp2[1] = seta_leeward;
        tmp2[2] = seta_windward;
        tmp2[3] = h_leeward;
        tmp2[4] = h_windward;
        tmp2[5] = hwall;

        tmp2[6] = R0;
        tmp2[7] = x_inlet;
        tmp2[8] = x_end;
        tmp2[9] = alfax_buff;
        tmp2[10] = seta_comput_domain;
    }

    MPI_Bcast(tmp1, 7, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmp2, 11, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        nx = tmp1[0];
        ny = tmp1[1];
        nz = tmp1[2];

        N = tmp1[3];

        nx_buff = tmp1[4];
        Iflag_mesh_seta = tmp1[5];
        ngrid_buffer = tmp1[6];


        seta_cone = tmp2[0];
        seta_leeward = tmp2[1];
        seta_windward = tmp2[2];
        h_leeward = tmp2[3];
        h_windward = tmp2[4];
        hwall = tmp2[5];

        R0 = tmp2[6];
        x_inlet = tmp2[7];
        x_end = tmp2[8];
        alfax_buff = tmp2[9];
        seta_comput_domain = tmp2[10];
    }

    seta_cone = seta_cone * PI / 180.;
    seta_leeward = seta_leeward * PI / 180.;
    seta_windward = seta_windward * PI / 180.;
    seta_comput_domain = seta_comput_domain * PI / 180.;

    NZ = nz/n_processe;

    if(my_id < nz%n_processe) NZ += 1;

    NPZ = (int*)malloc(n_processe * sizeof(int));
    NP = (int*)malloc(n_processe * sizeof(int));

    memset((void*)NPZ, 0, n_processe*sizeof(int));
    memset((void*)NP, 0, n_processe*sizeof(int));

    for(int i = 0; i < n_processe; i++){
        if(i < nz%n_processe){
            NPZ[i] = (int)nz/n_processe + 1;
        }else{
            NPZ[i] = (int)nz/n_processe;
        }
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];
    }

    if(NP[n_processe-1] != nz-NPZ[n_processe-1]) printf("NP is wrong![debug]\n");

}

#define Malloc_Judge(Memory)\
    {   if(Memory == NULL){\
        printf("Memory allocate error ! Can not allocate enough momory !!!\n");\
        exit(0); }\
    }

void Data_malloc(){
    x3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(x3d);

    y3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(y3d);

    z3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(z3d); 
}
#undef Malloc_Judge

void get_seta_mesh(double Useta[] ){ 
    double hh;
    //double seta1d[ny];
    char filename[120];  

    if (Iflag_mesh_seta == 0){ 
        hh = 2.0 * PI / ny;
        if(my_id == 0) printf("0 hh = %lf\n",hh);
        for (int j = 0; j < ny; j++){
            Useta[j] = j * hh;
        }
    }
    else if (Iflag_mesh_seta == 1){
        hh = PI / (ny - 1.0);
        if(my_id == 0) printf("1 hh = %lf\n",hh);
        for (int j = 0; j < ny; j++){
            Useta[j] = j * hh;
        }
    }
    else get_seta_new( Useta );
    
    sprintf(filename, "seta1d.dat");
    if(my_id == 0){
        printf("bluntcone mesh in seta direction is generating...\n");
        fp = fopen(filename, "w");
        for(int i = 0; i < ny; i++){
            fprintf(fp, "%d%15.6f\n", i, Useta[i] * 180.0 / PI);
        }
        fclose(fp);
    } 
}

void get_seta_new( double yseta1[]){
    
    char filename[120];
    double *yseta;
    double *ys1,*ys2,*ys3;
/*==========================================================================================================================
    zone 1: computation mesh; zone 2 and zone 3: buffer mesh
    zone 1 uniform dense mesh; zone 3 uniform rare mesh; zone 2 mesh between zone 1 and zone 3 (using Hermite interpolation)
    seta_center=PI/2.d0 均匀密网格以90°为中心左右对称
===========================================================================================================================*/
    double domain_factor = 0.7;  // length of zone 3= domain_factor*(zone 2+ zone 3)
    double grid_factor = 0.5;  // mesh of zone 3 =mesh_factor*(zone 2 + zone 3)

    if(ny%2 != 0){
        if(my_id == 0) printf("The mesh number in seta-direction should be an even number !!!\n");
        MPI_Finalize();
        exit(0);
    }

    //  ny1=(nseta-nseta1+1)/2
    //  ny2=(nseta1+1)/2
    double Seta_L1 = seta_comput_domain / 2.0;
    double Seta_L3 = (PI - Seta_L1) * domain_factor;
    int ny0 = ny / 2 + 1;
    int ny1 = (ny - ngrid_buffer + 1) / 2;
    int ny3 = (ny0 - ny1) * grid_factor + 0.5; //ny3 = (ny0 - ny1) * grid_factor + 0.5;
    int ny2 = ny0 + 2 - ny1 - ny3;

    if(my_id == 0) printf("seta-mesh: ny1=%d,ny2=%d,ny3=%d\n",ny1,ny2,ny3);

    ys1 = (double*)malloc(ny1 * sizeof(double));
    ys2 = (double*)malloc(ny2 * sizeof(double));
    ys3 = (double*)malloc(ny3 * sizeof(double));
    yseta = (double*)malloc(ny * sizeof(double));

    memset((void*)ys1, 0, ny1);
    memset((void*)ys2, 0, ny2);
    memset((void*)ys3, 0, ny3);
    memset((void*)yseta, 0, ny);
    //double ys1[ny1], ys2[ny2], ys3[ny3];
    //=========================== mesh in zone 1 ==============================
    double hy1 = Seta_L1 / (ny1 - 1.0);
    for(int i = 0; i < ny1; i++){
         ys1[i] = i * hy1;
         //if(my_id == 0) printf("i = %d, ys1 = %lf\n",i, ys1[i]);
    }
    //=========================== mesh in zone 3 ==============================
    double hy2 = Seta_L3 / (ny3 - 1.0);
    for(int i = 0; i < ny3; i++){
         ys3[i] = i * hy2 + (PI - Seta_L3);
         //if(my_id == 0) printf("i = %d, ys3 = %lf\n",i, ys3[i]);
    }
    //=========================== mesh in zone 2 ==============================
    double hs = 1.0 / (ny2 - 1.0);
    double ya = ys1[ny1-1];
    double yb = ys3[0];
    double ysa = (ys1[ny1-1] - ys1[ny1-2]) / hs;
    double ysb = (ys3[1] - ys3[0]) / hs;

    for(int i = 0; i < ny2; i++){
        double s = i * hs;
        double a1 = (2.0 * s + 1.0) * pow((s - 1.0), 2);
        double a2 = (3.0 - 2.0 * s) * s * s;
        double a3 = s * pow((s - 1.0), 2 );
        double a4 = s * s * (s - 1.0);
        ys2[i] = ya * a1 + yb * a2 + ysa * a3 + ysb * a4;  // Hermite interpolation
        //if(my_id == 0) printf("i = %d, ys2 = %lf\n",i, ys2[i]);
    }
    //=========================================================================
    for(int i = 0; i < ny1; i++){  //ny1个
        yseta[i] = ys1[i];
        //if(my_id == 0) printf("(ny1) i = %d, yseta = %lf, i0 = %d\n",i, yseta[i], i);
    }
    for(int i = 1; i < ny2; i++){  //ny2-1个
        yseta[i + ny1 -1] = ys2[i];
        //if(my_id == 0) printf("(ny2) i = %d, yseta = %lf, i0 = %d\n",i + ny1 -1, yseta[i + ny1 -1], i);
    }
    for(int i = 1; i < ny3; i++){  //ny3-1个
        yseta[i + ny1 + ny2 - 2] = ys3[i];
        //if(my_id == 0) printf("(ny3) i = %d, yseta = %lf, i0 = %d\n",i + ny1 + ny2 - 2, yseta[i + ny1 + ny2 - 2], i);
    }
    for(int i = ny0; i< ny; i++){
        yseta[i] = 2.0 * PI - yseta[2 * ny0 - (i+2)];  //可能有问题 yseta[i] = 2.0 * PI - yseta[2 * ny0 - (i+1)];
        //if(my_id == 0) printf("(ny0->ny) i = %d, yseta = %lf\n",i, yseta[i]);
    }
    //=============================Rote the grid=================================
    double seta0 = PI / 2.0;
    for(int i = 0; i < ny; i++){
        int i1 = i + 1 + ny/2;                    
        if(i1 > ny) i1 = i1 - ny;
        yseta1[i] = yseta[i1] + seta0; 
        
        if(yseta1[i] < 0) yseta1[i] = yseta1[i] + 2.0*PI;
        if(yseta1[i] >= 2.0*PI) yseta1[i] = yseta1[i] - 2.0*PI;
    }
    //=================== find the minimum seta, and set to zero=====================
    int i0 = 1;
    double cmax = 0.0;
    for(int i = 0; i < ny; i++){
        if(cos(yseta1[i]) > cmax){
            cmax = cos(yseta1[i]);
            i0 = i;
        }
    }

    seta0 = yseta1[i0];
    if(my_id == 0) printf("i0 (seta=0) = %d\n",i0);

    for(int i = 0; i < ny; i++){
        yseta1[i] = yseta1[i] - seta0;
        if(yseta1[i] < 0) yseta1[i] = yseta1[i] + 2.0 * PI;
        if(yseta[i] >= 2.0 * PI) yseta1[i] = yseta1[i] - 2.0 * PI;
    }
    //===================== find the maximum seta ===================================
    i0 = 1;      
    double cmin = 0.0;
    for(int i = 0; i < ny; i++){
        if(cos(yseta1[i]) < cmin){
            cmin = cos(yseta1[i]);
            i0 = i;
        }
    }

    printf("my_id = %d,n_processe = %d\n",my_id,n_processe);

    sprintf(filename, "seta.dat");
    if(my_id == 0){
        printf("i0 (seta=180) = %d\n",i0);
        fp = fopen(filename, "w");
        for(int i = 0; i < ny; i++){
            fprintf(fp, "%d%15.6f\n", i, yseta1[i]);
        }
        fclose(fp);
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Comput grid in seta direction is OK!\n");
}

void getsx(double dsx[]){
    char filename[120];
    //double sx[nx];
    double dx = 1.0 /(nx - nx_buff - 1);
    for(int i = 0; i < (nx - nx_buff); i++){
        dsx[i] = i * dx;
    }
    for(int i = (nx - nx_buff); i < nx; i++){
        dsx[i] = dsx[i-1] + alfax_buff * (dsx[i-1] - dsx[i-2]);
    }

    sprintf(filename, "sx.dat");
    if(my_id == 0){
        printf("write sx.dat ...\n");
        fp = fopen(filename, "w");
        for(int i = 0; i < nx; i++){
            fprintf(fp, "%d%15.6f\n", i, dsx[i]);
        }
    
        fclose(fp);

    }
}

void generate_mesh(double Useta1[],double Udsx[]){
    double seta,sk;
    double RR=0.0;
    double sy[ny];
    double x0,y0;
    double xa,ya,xb,yb;
    double x1,y1,x2,y2,x3,y3,x4,y4;
    double seta_upper;
    double SL;

    /*x3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    y3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    z3d = (double*)malloc(nx * ny * NZ * sizeof(double));*/

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    for(int j = 0; j < ny; j++){
        seta = Useta1[j];
        if(my_id == 0) printf("j = %d, seta = %lf\n", j, seta * 180.0 / PI);
        sk = seta / PI;
        if(sk > 1.0) sk = 2.0 - sk;
        
        
        //======================= (x1,y1) lift-bottom point ===========================
        x1 = x_inlet;
        y1 = R0 * cos(seta_cone) + (x1 + R0 * sin(seta_cone)) * tan(seta_cone);
        
        //======================= (x2,y2) right-bottom point ===========================
        x2 = x_end;
        y2 = y1 + (x2 - x1) * tan(seta_cone);
        
        //    r1=h_leeward+sk*(h_windward-h_leeward)  <====  hight of wall-normail domain
        RR = Rh_hermit( RR,sk);  // Hermit interpolation
        
        seta_upper = seta_leeward + sk * (seta_windward - seta_leeward); 
        //if(my_id == 0) printf("j = %d, sk = %lf, RR = %lf\n", j, sk, RR);
        //======================= (x3,y3) lift-upper point =============================
        x3 = x1 + RR * cos(seta_cone + PI / 2.0);
        y3 = y1 + RR * sin(seta_cone + PI / 2.0);

        // The computation domain is the qudrangle (x1,y1)-(x2,y2)-(x4,y4)-(x3,y3)
        //======================= (x4,y4) right-upper point =============================
        x4 = (y2 - y3 + x3 * tan(seta_upper) + x2 /tan(seta_cone)) / (tan(seta_upper) + 1.0 / (seta_cone));
        y4 = y3 + (x4 - x3) * tan(seta_upper);

        //if(my_id == 0) printf("x1 = %lf, y1 = %lf\nx2 = %lf, y2 = %lf\nx3 = %lf, y3 = %lf\nx4 = %lf, y4 = %lf\n", x1, y1, x2, y2, x3, y3, x4, y4);

        for(int i = 0; i < nx; i++){
            xa = x1 + Udsx[i] * (x2 - x1);
            ya = y1 + Udsx[i] * (y2 - y1);
            xb = x3 + Udsx[i] * (x4 - x3);
            yb = y3 + Udsx[i] * (y4 - y3);

            SL = sqrt( pow((xa - xb), 2) + pow((ya - yb), 2) );
            //if(my_id == 0) printf(" i = %d, j = %d, SL = %lf\n",i,j,SL); 

            getsy(sy,SL,hwall);
            

            for(int k = 0; k < NZ; k++){
            //if(my_id == 0) printf(" i = %d, j = %d, k = %d, sy = %lf\n",i,j,k,sy[k]);    
                x0 = xa + (xb - xa) * sy[k];
                y0 = ya + (yb - ya) * sy[k];

                xx3d[k][j][i] = x0;
                yy3d[k][j][i] = y0 * sin(seta);
                zz3d[k][j][i] = y0 * cos(seta);

            }
        }

    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Comput gridxyz is OK!\n");
}

void getsy(double sy[], double SL, double deltx){
    char filename[120];
    double fb,fbx,bnew,a,s;
    double b = 3.5;  // sy=a(exp(b*s)-1)
    double delta = deltx / SL;
    double dy = 1.0 / (nz - 1.0);
    
    double *sy_buff;

    sy_buff = (double*)malloc( NZ * sizeof(double));

    double (*ssy_buff) = (double(*))sy_buff;
    //  using Newton method to get coefficient

    label:  //标记label标签
    fb = (exp(b / (nz - 1.0)) - 1.0) / (exp(b) - 1.0) - delta;
    fbx = (exp(b / (nz - 1.0))/(nz - 1.0) * (exp(b)- 1.0) - (exp(b/(nz-1.0))-1.0)*exp(b) ) / pow((exp(b)-1.0), 2);

    bnew = b - fb / fbx;

    if(fabs(b - bnew) > 1.0e-6){
        b = bnew;
        goto label;
    }

    b = bnew;
    a = 1.0 / (exp(b) - 1.0);
    //for(int n = 0; n < n_processe; n++){    //for(int j = 0; j < nz; j++){
        for(int j = 0; j < NZ; j++){
            s =(double)(j + NP[my_id]) * dy;      //应该是有问题要改的!!!!  NP[my_id]?
            sy[j] = a * (exp(s * b) - 1.0);
            ssy_buff[j] = sy[j];           //设置缓冲数组防止数据交换时覆盖掉
        } 
    //}

    /*sprintf(filename, "sy.dat");
    
    if(my_id == 0){
        printf("write sy.dat ...\n");

        fp = fopen(filename, "w");
        fprintf(fp, "variables=k,sy\n");
    
        fclose(fp);
    }

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                        
                            fprintf(fp, "%d%15.6f\n", k, ssy_buff[k]);
                }

                fclose(fp);


            }    
    
            if(my_id != 0){
                MPI_Send(sy_buff, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(sy_buff, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }*/


}

double Rh_hermit(double Rh, double x ){   //Rh_hermit(R1, sk);  // Hermit interpolation
    double Ah0,Ah1,x1,xp;
    xp = 4.0 / 5.0;
    if (x >= xp){
        Rh = h_windward;
        return Rh;
    }
    else{
        x1 = x / xp;
        Ah0 = (1.0 + 2.0 * x1) * pow((x1 - 1.0), 2);   //3rd Hermit
        Ah1 = (1.0 - 2.0 * (x1 - 1.0)) * x1 * x1;
        //  Sh0=x*(x-1.d0)**2
        //  Sh1=(x-1.d0)*x*x
        Rh = h_leeward * Ah0 + h_windward * Ah1;
    }
    //if(my_id == 0) printf(" sk = %lf, R1 = %lf\n",  x, Rh);
    return Rh;
}

void output_grid_i0(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x3d_buff, *y3d_buff, *z3d_buff;

    x3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny] = (double(*)[ny])x3d_buff;
    double (*yy3d_buff)[ny] = (double(*)[ny])y3d_buff;
    double (*zz3d_buff)[ny] = (double(*)[ny])z3d_buff;

    char filename[120];

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            xx3d_buff[k][j] = xx3d[k][j][i0];      //设置缓冲数组防止数据交换时覆盖掉
            yy3d_buff[k][j] = yy3d[k][j][i0];
            zz3d_buff[k][j] = zz3d[k][j][i0];    

        }
    }

    sprintf(filename, "gridi0.dat");
    
    if(my_id == 0){
        printf("write gridi0 data ...\n");

        fp = fopen(filename, "w");
        fprintf(fp, "variables=x,y,z\n");
        fprintf(fp, "zone i=%d ,j=%d, k=%d\n", 1, ny, nz);
    
        fclose(fp);
    }

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", 
                            xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j]);
                        
                    }
                }

                fclose(fp);


            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }
        

}

void output_grid_j0(int j0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x3d_buff, *y3d_buff, *z3d_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;

    char filename[120];

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < nx; j++){
            xx3d_buff[k][j] = xx3d[k][j0][j];      //设置缓冲数组防止数据交换时覆盖掉
            yy3d_buff[k][j] = yy3d[k][j0][j];
            zz3d_buff[k][j] = zz3d[k][j0][j];    

        }
    }

    sprintf(filename, "gridj0.dat");
    
    if(my_id == 0){
        printf("write gridj0 data ...\n");

        fp = fopen(filename, "w");
        fprintf(fp, "variables=x,y,z\n");
        fprintf(fp, "zone i=%d ,j=%d, k=%d\n", nx, 1, nz);
    
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
        

}

void output_mesh3d(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file); //MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id == 0) printf("Write X3d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, x3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("Write Y3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, y3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("Write Z3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, z3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);

}

void Finalize(){
    free(x3d);
    free(y3d);
    free(z3d);

}