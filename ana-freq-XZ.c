
//对流场保存的截面时间序列数据 eg：Savedata-XZ00?.dat 进行傅里叶变换得到频谱(有量纲 spectrum-?.dat 和无量纲 findspectrum-?.dat )
//保存时间平均后的截面信息 eg：XZplane-time-average-?.dat 方便进行展向空间关联
//保存一个点的物理量随时间(无量纲)的变化序列 eg：time-data-?.dat
//根据无量纲的频谱挑选需要的频率进行傅里叶变换（需要在运行一次程序），得到该频率在流场中的空间分布 eg：spectrum2d-?.dat
//输出所提取无量纲频率的型函数

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>   //复数头文件

#include "mpi.h"
#include "pthread.h"
  
//============================需要修改的参数==============================
#define PI 3.141592653589793                                              
#define mp 5000     //100个时间序列，可以按需求更改 
#define xposition 5000  //用于输出流向对应位置的时间序列、频谱信息等数据  
#define k0 1                       
//#define np 0       //第几个YZ截面，对应程序调用中的i0
int nxpoints[] = {896,1152,1408,1664,1920,2176,2432,2688,2944,3200,3456,3712,3968,4480,5120,5760};  //x对应的编号
int nypoints[] = {0,5800};  //y对应的编号   
//还有DMD.in文件    
int wk1[] = {244,237,224,219,198,188,170};   //转捩前幅值最大的无量纲频率
int wk2[] = {400,374};
int m0=0, m1=20, m2=35;  //m0:壁面    m1:线性区和对数区的交界处     m2:对数区
//========================================================================

FILE *fp;
FILE *fp1;
FILE *fp2;
FILE *fp3;

MPI_Status status;

char  str[2000];
int init[3];
int my_id, n_processe;
int nx, ny, nz, NZ, *NPZ, *NP, *head,*NPE;
int NZk, *NPZk, *NPk;
int np, ib, ie, kb, ke, mp_end, nstep;
int nz_myid;
int ip1, kp1, ip2, kp2, ip3, kp3, ip4, kp4;
int ni, nk;
double Re, Ama, Gamma, Pr, T_Ref, Tw, Amu, Cp, hh, tmp;
double p00;
double tmp1, tmp2,tmp3, tmp4, tmp5, tmp6, tmp7;
double *x3d, *y3d, *z3d;
double *pd, *pu, *pv, *pw, *pT;
double *pd1, *pu1, *pv1, *pw1, *pT1 ,*pP1;
double *pd0, *pu0, *pv0, *pw0, *pT0 ,*pP0;
double *pdrms, *purms, *pvrms, *pwrms, *pTrms ,*pPrms;
double *pd2, *pu2, *pv2, *pw2, *pT2 ,*pP2;
double *dtmp, *utmp, *vtmp, *wtmp, *Ttmp, *Ptmp;

double tt[mp];  //读取mp个时间序列，tt存放无量纲时间
double P1d0[mp], P1d1[mp], P1d2[mp]; //0对应壁面，1对应边界层附近，2对应边界层外缘
double T1d0[mp], T1d1[mp], T1d2[mp];
double d1d0[mp], d1d1[mp], d1d2[mp];
double u1d0[mp], u1d1[mp], u1d2[mp];
double T1d[2][mp];
double d1d[2][mp];
double u1d[2][mp];
double deltx = 2.0 * PI / mp;
double dt0 = 0.10 * (9.35376e-7);
//double wk = 96.0; //做二维傅里叶变换时的无量纲频率
char fP[10] = "P";
char fT[10] = "T";
char fd[10] = "d";
char fu[10] = "u";

void mpi_init(int *Argc, char ***Argv);
void Data_malloc();
void Read_parameter();
void Read_mesh();
void Read_Savedata(int i0);
void flow1d_tt(int m, int j, double U);    //目前还不好用，可以进行修改优化
void Write_Savedata_timeaverage_XZ(int i0);  
void Compute_frequancy_XZ(double U1d[], int i0, int i1, char* Uf ); //i0为选取的YZ截面编号，i1用于区分输出的物理量及其位置，i1=0为壁面处压力，i1=1为边界层附近压力
void Compute_frequancy_2dYZ( int i0, int wk ); //wk:做二维傅里叶变换时的无量纲频率
void Compute_frequancy_2dXZ( int i0, int wk );
void get_shapefunction(int i0,int xx,double Ud2[][nx],double Uu2[][nx],double Uv2[][nx],double Uw2[][nx],double UT2[][nx],double UP2[][nx], int wk);
void Read_data();
void Write_data_new();
void Write_data_new_extension();
void Read_mesh_1d();
void Read_data_1d();
void Write_grid_format();
void Write_data_format();
void Write_dataxy2d_first_format();
void Write_dataxy2d_first_format_total();
void Write_datayz2d_format();
void Write_dataxz2d_format();
void Write_datayz2d1_format();
void Write_dataxy2d1_format();
void Write_dataxyz3d_format();
void Write_flow1d_inlet();
void Write_cf2d_format();
void Write_cf3d_format();
void Write_cf3d_format_total();
void Write_dataxz2d_cf_double_cone(int i0);
void Write_dataxy2d_cf_compress_ramp(int i0);
//void Write_dataxy2d_yp();
void DMD_XZ_coordinate(int i0);
void Compute_Raa(int i0);
void Finalize();

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();
    Data_malloc();

    Read_mesh();

//====================调用函数的时候需要修改参数=======================
    
        Read_Savedata(np);                //读Savedata-YZ000.dat         
        Write_Savedata_timeaverage_XZ(np);
        //DMD_XZ_coordinate(np);
        /*Compute_frequancy_XZ(P1d0, np, 0, fP);
        Compute_frequancy_XZ(P1d1, np, 1, fP);
        Compute_frequancy_XZ(P1d2, np, 2, fP);
        Compute_frequancy_XZ(T1d0, np, 0, fT);
        Compute_frequancy_XZ(T1d1, np, 1, fT);
        Compute_frequancy_XZ(T1d2, np, 2, fT);
        Compute_frequancy_XZ(d1d0, np, 0, fd);
        Compute_frequancy_XZ(d1d1, np, 1, fd);
        Compute_frequancy_XZ(d1d2, np, 2, fd);
        Compute_frequancy_XZ(u1d0, np, 0, fu);
        Compute_frequancy_XZ(u1d1, np, 1, fu);
        Compute_frequancy_XZ(u1d2, np, 2, fu);*/
        /*Compute_frequancy_2dXZ( np , wk1[0]); //第二个参数为无量纲频率wk
        Compute_frequancy_2dXZ( np , wk1[1]);
        Compute_frequancy_2dXZ( np , wk1[2]);
        Compute_frequancy_2dXZ( np , wk1[3]);
        Compute_frequancy_2dXZ( np , wk1[4]);
        Compute_frequancy_2dXZ( np , wk1[5]);
        Compute_frequancy_2dXZ( np , wk1[6]);
        Compute_frequancy_2dXZ( np , wk2[0]);
        Compute_frequancy_2dXZ( np , wk2[1]);*/
        Compute_Raa(np);

//=====================================================================

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
        if((fp = fopen("opencfd-scu.in", "r")) == NULL){
            printf("Can't open this file: 'opencfd-scu.in'\n");
            exit(0);
        }

        fgets(str, 2000, fp);
        fgets(str, 2000, fp);
        fscanf(fp, "%d%d%d\n", &nx,&ny,&nz);

        for(int i=0; i<11; i++){
            fgets(str, 2000, fp);
        }
    
        fscanf(fp, "%lf%lf%lf%lf%lf%lf\n", &Re, &Ama, &Gamma, &Pr, &T_Ref, &tmp);
        printf("Computation start...\nRe is %lf\nAma is %lf\nGamma is %lf\nPr is %lf\nT_Ref is %lf\ntmp is %lf\n",
              Re, Ama, Gamma, Pr, T_Ref, tmp);
   
        for(int i=0; i<3; i++){
            fgets(str, 2000, fp);
        }
    
        /*for(int i=0; i<7; i++){ 
            fscanf(fp, "%lf", &tmp);
            printf("tmp is %lf\n", tmp); 
        }*/

        fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf\n", &tmp1, &tmp2, &tmp3, &tmp4, &tmp5, &tmp6, &tmp7, &Tw);
        printf("tmp1 is %lf\ntmp2 is %lf\ntmp3 is %lf\ntmp4 is %lf\ntmp5 is %lf\ntmp6 is %lf\ntmp7 is %lf\nTw is %lf\n", 
               tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, Tw);

        fclose(fp);

        if((fp3 = fopen("DMD-data.in", "r")) == NULL){
            printf("Can't open this file: 'DMD-data.in'\n");
            exit(0);
        }
        fgets(str, 2000, fp3);
        fgets(str, 2000, fp3);
        fscanf(fp3, "%d%d%d%d%d%d%d%d\n", &np,&ib,&ie,&kb,&ke,&mp_end,&nstep,&nz_myid);
        printf("np is %d\nib is %d\nie is %d\nkb is %d\nke is %d\nmp_end is %d\nnstep is %d\nnz_myid is %d\n",
              np, ib, ie, kb, ke, mp_end, nstep, nz_myid);
        fgets(str, 2000, fp3);
        fscanf(fp3, "%d%d%d%d%d%d%d%d\n", &ip1,&kp1,&ip2,&kp2,&ip3,&kp3,&ip4,&kp4);
        printf("ip1 is %d\nkp1 is %d\nip2 is %d\nkp2 is %d\nip3 is %d\nkp3 is %d\nip4 is %d\nkp4 is %d\n",
              ip1, kp1, ip2, kp2, ip3, kp3, ip4, kp4);
        fclose(fp3);
    }
//-----------------------------------------------------------------------------------------------------
    int tmp1[17];
    double tmp2[9];

    if(my_id == 0){
        tmp1[0] = nx;
        tmp1[1] = ny;
        tmp1[2] = nz;

        tmp2[0] = Re;
        tmp2[1] = Ama;
        tmp2[2] = Gamma;
        tmp2[3] = Pr;

        tmp2[4] = T_Ref;
        tmp2[5] = Tw;

        tmp1[3] = ib;
        tmp1[4] = ie;
        tmp1[5] = kb;
        tmp1[6] = ke;
        tmp1[7] = mp_end;
        tmp1[8] = nstep;
        tmp1[9] = ip1;
        tmp1[10] = kp1;
        tmp1[11] = ip2;
        tmp1[12] = kp2;
        tmp1[13] = ip3;
        tmp1[14] = kp3;
        tmp1[15] = ip4;
        tmp1[16] = kp4;
    }

    MPI_Bcast(tmp1, 17, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmp2, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        nx = tmp1[0];
        ny = tmp1[1];
        nz = tmp1[2];

        Re = tmp2[0];
        Ama = tmp2[1];
        Gamma = tmp2[2];
        Pr = tmp2[3];

        T_Ref = tmp2[4];
        Tw = tmp2[5];

        ib = tmp1[3];
        ie = tmp1[4];
        kb = tmp1[5];
        ke = tmp1[6];
        mp_end = tmp1[7];
        nstep = tmp1[8];
        ip1 = tmp1[9];
        kp1 = tmp1[10];
        ip2 = tmp1[11];
        kp2 = tmp1[12];
        ip3 = tmp1[13];
        kp3 = tmp1[14];
        ip4 = tmp1[15];
        kp4 = tmp1[16];
    }
//-------------------------------------------------------------------------------------------

    Amu = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(Tw*Tw*Tw)/(110.4/T_Ref + Tw);
    Cp = 1.0/((Gamma - 1)*Ama*Ama);   /*原来Cp = Gamma/((Gamma - 1)*Ama*Ama);*/
    p00 = 1.0/(Gamma*Ama*Ama);

    ni = ie - ib + 1;
    nk = ke - kb + 1;

//-------------------------------------------------------------------------------------------

    NZ = nz/n_processe;      //沿壁面法向划分多块
    NZk = nk/n_processe;

    if(my_id < nz%n_processe) NZ += 1;
    if(my_id < nk%n_processe) NZk += 1;

    NPZ = (int*)malloc(n_processe * sizeof(int));
    NP = (int*)malloc(n_processe * sizeof(int));
    NPE = (int*)malloc(n_processe * sizeof(int));
    NPZk = (int*)malloc(n_processe * sizeof(int));
    NPk = (int*)malloc(n_processe * sizeof(int));

    memset((void*)NPZ, 0, n_processe*sizeof(int));   //为malloc新申请的连续内存进行初始化
    memset((void*)NP, 0, n_processe*sizeof(int));
    memset((void*)NPZk, 0, n_processe*sizeof(int));   
    memset((void*)NPk, 0, n_processe*sizeof(int));
    memset((void*)NPE, 0, n_processe*sizeof(int));

    for(int i = 0; i < n_processe; i++){
        if(i < nz%n_processe){
            NPZ[i] = (int)nz/n_processe + 1;
        }else{
            NPZ[i] = (int)nz/n_processe;
        }
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];  //偏移 
        NPE[0] = NPZ[0];
        if(i != 0) NPE[i] = NPE[i-1] + NPZ[i];   

        if(i < nk%n_processe){
            NPZk[i] = (int)nk/n_processe + 1;
        }else{
            NPZk[i] = (int)nk/n_processe;
        }
        NPk[0] = 0;
        if(i != 0) NPk[i] = NPk[i-1] + NPZk[i-1];  //偏移  
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

    //x2d = (double*)malloc(ni * NZk * sizeof(double));
    //Malloc_Judge(x2d);


    head = (int*)malloc(5 * sizeof(int));

    dtmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(dtmp);

    utmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(utmp);
    
    vtmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(vtmp);

    wtmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(wtmp);

    Ttmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(Ttmp);

    Ptmp = (double*)malloc( mp * nx * NZ * sizeof(double));
    Malloc_Judge(Ptmp);


    pd1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pd1);

    pu1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pu1);

    pv1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pv1);

    pw1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pw1);

    pT1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pT1);

    pP1 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pP1);

    pd0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pd0);
    memset((void*)pd0, 0, nx * NZ * sizeof(double));

    pu0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pu0);
    memset((void*)pu0, 0, nx * NZ * sizeof(double));

    pv0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pv0);
    memset((void*)pv0, 0, nx * NZ * sizeof(double));

    pw0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pw0);
    memset((void*)pw0, 0, nx * NZ * sizeof(double));

    pT0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pT0);
    memset((void*)pT0, 0, nx * NZ * sizeof(double));

    pP0 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pP0);
    memset((void*)pP0, 0, nx * NZ * sizeof(double));

    pdrms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pdrms);
    memset((void*)pdrms, 0, nx * NZ * sizeof(double));

    purms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(purms);
    memset((void*)purms, 0, nx * NZ * sizeof(double));

    pvrms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pvrms);
    memset((void*)pvrms, 0, nx * NZ * sizeof(double));

    pwrms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pwrms);
    memset((void*)pwrms, 0, nx * NZ * sizeof(double));

    pTrms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pTrms);
    memset((void*)pTrms, 0, nx * NZ * sizeof(double));

    pPrms = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pPrms);
    memset((void*)pPrms, 0, nx * NZ * sizeof(double));

    pd2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pd2);
    memset((void*)pd2, 0, nx * NZ * sizeof(double));

    pu2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pu2);
    memset((void*)pu2, 0, nx * NZ * sizeof(double));

    pv2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pv2);
    memset((void*)pv2, 0, nx * NZ * sizeof(double));

    pw2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pw2);
    memset((void*)pw2, 0, nx * NZ * sizeof(double));

    pT2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pT2);
    memset((void*)pT2, 0, nx * NZ * sizeof(double));

    pP2 = (double*)malloc( nx * NZ * sizeof(double));
    Malloc_Judge(pP2);
    memset((void*)pP2, 0, nx * NZ * sizeof(double));

}

#undef Malloc_Judge

#define FREAD(ptr , size , num , stream) \
    {   int tmp_buffer;\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
        fread(ptr , size , num , stream);\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
    }

void Read_mesh_1d(){
    if(my_id == 0){
        if((fp = fopen("OCFD3d-Mesh.dat", "r")) == NULL){
            printf("Can't open this file: 'OCFD3d-Mesh.dat'\n");
            exit(0);
        }

        int num = nx * ny;
        printf("Read OCFD3d-Mesh.dat-X ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(x3d + num * k, sizeof(double), num, fp);
        }

        printf("Read OCFD3d-Mesh.dat-Y ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(y3d + num * k, sizeof(double), num, fp);
        }

        printf("Read OCFD3d-Mesh.dat-Z ...\n\n");

        for(int k = 0; k < 2; k++){
            FREAD(z3d + num * k, sizeof(double), num, fp);
        }

        fclose(fp);
    }
}

void Read_data_1d(){
    if(my_id == 0){
        if((fp = fopen("opencfd.ana", "r")) == NULL){
            printf("Can't open this file: 'opencfd.ana'\n");
            exit(0);
        }

        int num = nx * ny;
        printf("Read opencfd.ana-d ...\n");

        fread(head , sizeof(int) , 5 , fp);

        for(int k = 0; k < 2; k++){
            FREAD(pd + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-u ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pu + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-v ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pv + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-w ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pw + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-T ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pT + num * k, sizeof(double), num, fp);
        }

        fclose(fp);
    }
}

#undef FREAD

void Write_flow1d_inlet(){
    double tmp[ny][4];

    if(my_id == 0){
            char str[100];
            double tmp1;
            fp = fopen("flow1d-inlet.dat", "r");
            printf("read inlet boundary data: flow-inlet.dat\n");
            fgets(str, 100, fp);
            for(int j = 0; j < ny; j++){
                fscanf(fp, "%lf%lf%lf%lf%lf\n", &tmp1, &tmp[j][0], &tmp[j][1], &tmp[j][2], &tmp[j][3]);
            }
            fclose(fp);

            fp = fopen("flow1d-inlet.dat", "w");
            fscanf(fp, "variables=y, d, u, v, T\n");
            for(int j = 0; j < ny-1; j++){
                fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[j][0], tmp[j][1], tmp[j][2], tmp[j][3]);
                fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, (tmp[j][0]+tmp[j+1][1])/2, (tmp[j][1]+tmp[j+1][1])/2, (tmp[j][2]+tmp[j+1][2])/2, (tmp[j][3]+tmp[j+1][3])/2);
            }

            fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[ny-1][0], tmp[ny-1][1], tmp[ny-1][2], tmp[ny-1][3]);
            fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[ny-1][0], tmp[ny-1][1], tmp[ny-1][2], tmp[ny-1][3]);
    }
}

void Read_mesh(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id == 0) printf("READ X3d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, x3d+num*k, num,  MPI_DOUBLE, &status);    //偏移应该与并行的进程有关，MPI_File_read用法
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
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}

void Read_Savedata(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[nx] = (double (*)[nx])(pd1);
    double (*u)[nx] = (double (*)[nx])(pu1);
    double (*v)[nx] = (double (*)[nx])(pv1);
    double (*w)[nx] = (double (*)[nx])(pw1);
    double (*T)[nx] = (double (*)[nx])(pT1);
    double (*P)[nx] = (double (*)[nx])(pP1);

    double (*d0)[nx] = (double (*)[nx])(pd0);
    double (*u0)[nx] = (double (*)[nx])(pu0);
    double (*v0)[nx] = (double (*)[nx])(pv0);
    double (*w0)[nx] = (double (*)[nx])(pw0);
    double (*T0)[nx] = (double (*)[nx])(pT0);
    double (*P0)[nx] = (double (*)[nx])(pP0);

    double (*drms)[nx] = (double (*)[nx])(pdrms);
    double (*urms)[nx] = (double (*)[nx])(purms);
    double (*vrms)[nx] = (double (*)[nx])(pvrms);
    double (*wrms)[nx] = (double (*)[nx])(pwrms);
    double (*Trms)[nx] = (double (*)[nx])(pTrms);
    double (*Prms)[nx] = (double (*)[nx])(pPrms);

    double *x3d_buff, *y3d_buff, *z3d_buff;
    double *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff, *pP_buff;
    double *pd0_buff, *pu0_buff, *pv0_buff, *pw0_buff, *pT0_buff, *pP0_buff;
    double *pdrms_buff, *purms_buff, *pvrms_buff, *pwrms_buff, *pTrms_buff, *pPrms_buff;

    double *T2d, *P2d;

    T2d = (double*)malloc(ni * NZ * sizeof(double));
    P2d = (double*)malloc(ni * NZ * sizeof(double));

    double (*pT2d)[ni] = (double(*)[ni])(T2d);
    double (*pP2d)[ni] = (double(*)[ni])(P2d);

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pP_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pd0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pP0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pdrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    purms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pvrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pwrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pTrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pPrms_buff  = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd_buff)[nx]  = (double(*)[nx])pd_buff;
    double (*ppu_buff)[nx]  = (double(*)[nx])pu_buff;
    double (*ppv_buff)[nx]  = (double(*)[nx])pv_buff;
    double (*ppw_buff)[nx]  = (double(*)[nx])pw_buff;
    double (*ppT_buff)[nx]  = (double(*)[nx])pT_buff;
    double (*ppP_buff)[nx]  = (double(*)[nx])pP_buff;
    double (*ppd0_buff)[nx]  = (double(*)[nx])pd0_buff;
    double (*ppu0_buff)[nx]  = (double(*)[nx])pu0_buff;
    double (*ppv0_buff)[nx]  = (double(*)[nx])pv0_buff;
    double (*ppw0_buff)[nx]  = (double(*)[nx])pw0_buff;
    double (*ppT0_buff)[nx]  = (double(*)[nx])pT0_buff;
    double (*ppP0_buff)[nx]  = (double(*)[nx])pP0_buff;
    double (*ppdrms_buff)[nx]  = (double(*)[nx])pdrms_buff;
    double (*ppurms_buff)[nx]  = (double(*)[nx])purms_buff;
    double (*ppvrms_buff)[nx]  = (double(*)[nx])pvrms_buff;
    double (*ppwrms_buff)[nx]  = (double(*)[nx])pwrms_buff;
    double (*ppTrms_buff)[nx]  = (double(*)[nx])pTrms_buff;
    double (*ppPrms_buff)[nx]  = (double(*)[nx])pPrms_buff;
   
    int num =  nx;
    int num1 =  ni;
    int num_byte =  nx * sizeof(double);
    int num_byte1 =  ni * sizeof(double);
    int Istep;
    char fp_name[120];
    char fp_name1[120];
    char fp_name2[120];
    char filename[120];
    char filename1[120];
    char filename2[120];
    double tmpP1[mp], tmpT1[mp], tmpd1[mp], tmpu1[mp];
    double tmpP2[mp], tmpT2[mp], tmpd2[mp], tmpu2[mp];
    double tmpP0[mp];
    double tmp;

    MPI_File tmp_file;
    /*MPI_File tmp_file1;
    MPI_File tmp_file2;*/

    sprintf(fp_name, "Savedata-XZ%03d.dat", i0);
    /*sprintf(fp_name1, "DMD-data-XZ%03d-T.dat", i0);
    sprintf(fp_name2, "DMD-data-XZ%03d-P.dat", i0);*/

    MPI_File_open(MPI_COMM_WORLD, fp_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);
    /*MPI_File_open(MPI_COMM_WORLD, fp_name1, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file1);
    MPI_File_open(MPI_COMM_WORLD, fp_name2, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file2);*/

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
    /*MPI_File_seek(tmp_file1, 0, MPI_SEEK_SET);
    MPI_File_seek(tmp_file2, 0, MPI_SEEK_SET);*/

    for(int i = 0; i < mp; i++ ){                      //读取mp个时间序列

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	    MPI_File_read_all(tmp_file, &Istep, 1, MPI_INT, &status);             
        MPI_File_read_all(tmp_file, &tt[i], 1, MPI_DOUBLE, &status);
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        if(my_id == 0)printf("Istep=%05d, tt=%lf\n", Istep, tt[i]);



        if(my_id == 0) printf("READ Savedata d ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pd1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata u ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pu1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata v ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pv1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata w ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pw1+num*k, num,  MPI_DOUBLE, &status);
          
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata T ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)  ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pT1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR); 
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
    
        
        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < nx; j++){

                P[k][j]= d[k][j] * T[k][j];

                d0[k][j] = d0[k][j] + d[k][j];
                u0[k][j] = u0[k][j] + u[k][j];
                v0[k][j] = v0[k][j] + v[k][j];
                w0[k][j] = w0[k][j] + w[k][j];
                T0[k][j] = T0[k][j] + T[k][j];
                P0[k][j] = P0[k][j] + P[k][j];

                drms[k][j] = drms[k][j] + d[k][j] * d[k][j];
                urms[k][j] = urms[k][j] + u[k][j] * u[k][j];
                vrms[k][j] = vrms[k][j] + v[k][j] * v[k][j];
                wrms[k][j] = wrms[k][j] + w[k][j] * w[k][j];
                Trms[k][j] = Trms[k][j] + T[k][j] * T[k][j];
                Prms[k][j] = Prms[k][j] + P[k][j] * P[k][j];

                xx3d_buff[k][j] = xx3d[k][nypoints[i0]][j];      //设置缓冲数组防止数据交换时覆盖掉
                yy3d_buff[k][j] = yy3d[k][nypoints[i0]][j];
                zz3d_buff[k][j] = zz3d[k][nypoints[i0]][j];
                ppd_buff[k][j]  = d[k][j];
                ppu_buff[k][j]  = u[k][j];             
                ppv_buff[k][j]  = v[k][j];
                ppw_buff[k][j]  = w[k][j];
                ppT_buff[k][j]  = T[k][j];
                ppP_buff[k][j]  = P[k][j];
                ppd0_buff[k][j]  = d0[k][j];
                ppu0_buff[k][j]  = u0[k][j];
                ppv0_buff[k][j]  = v0[k][j];
                ppw0_buff[k][j]  = w0[k][j];
                ppT0_buff[k][j]  = T0[k][j];
                ppP0_buff[k][j]  = P0[k][j];
                ppdrms_buff[k][j]  = drms[k][j];
                ppurms_buff[k][j]  = urms[k][j];
                ppvrms_buff[k][j]  = vrms[k][j];
                ppwrms_buff[k][j]  = wrms[k][j];
                ppTrms_buff[k][j]  = Trms[k][j];
                ppPrms_buff[k][j]  = Prms[k][j];
                           
            }
        }
                
                if(my_id == 0){                    
                        P1d0[i] = P[0][xposition];        //k对应壁面法向位置，j对应流向位置，在对应my_id内赋值   
                        T1d0[i] = T[0][xposition]; 
                        d1d0[i] = d[0][xposition];  
                        u1d0[i] = u[0][xposition];                                            
                }
                
                int  n0, n1, n2, id0, id1, id2;   //m1 m2分别为壁面法向网格编号

                for(int i = 0; i < n_processe-1; i++){       //寻找m1,m2分别属于哪个my_id
                    if(NP[i]-1 < m1 && NP[i+1]-1 > m1){
                        id1 = i;
                    }
                    if(NP[i]-1 < m2 && NP[i+1]-1 > m2){
                        id2 = i;
                    }
                }                    
      
                n1 = m1 - NP[id1];                   //寻找所属my_id内的壁面法向编号
                n2 = m2 - NP[id2];
    
                if(my_id == id1){
                        tmpP1[i] = P[n1][xposition];        //在对应my_id内赋值 
                        tmpT1[i] = T[n1][xposition];
                        tmpd1[i] = d[n1][xposition];  
                        tmpu1[i] = u[n1][xposition];        
                }
                if(my_id == id2){
                        tmpP2[i] = P[n2][xposition];        //在对应my_id内赋值
                        tmpT2[i] = T[n2][xposition];
                        tmpd2[i] = d[n2][xposition];  
                        tmpu2[i] = u[n2][xposition];    
                }

                //MPI_Barrier(MPI_COMM_WORLD);         //阻断
                MPI_Bcast(tmpP1, mp, MPI_DOUBLE, id1, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(tmpT1, mp, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(tmpd1, mp, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                MPI_Bcast(tmpu1, mp, MPI_DOUBLE, id1, MPI_COMM_WORLD);
                
                MPI_Bcast(tmpP2, mp, MPI_DOUBLE, id2, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                MPI_Bcast(tmpT2, mp, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(tmpd2, mp, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                MPI_Bcast(tmpu2, mp, MPI_DOUBLE, id2, MPI_COMM_WORLD);
                
                if(my_id != id1){
                        P1d1[i] = tmpP1[i];         //在节点上赋值
                        T1d1[i] = tmpT1[i];
                        d1d1[i] = tmpd1[i];
                        u1d1[i] = tmpu1[i];
                }
                if(my_id != id2){
                        P1d2[i] = tmpP2[i];         //在节点上赋值
                        T1d2[i] = tmpT2[i];
                        d1d2[i] = tmpd2[i];
                        u1d2[i] = tmpu2[i];
                }

                //flow1d_tt(20, 180, P1d1[i]);

                //printf("P1d1[i] = %15.6f\n",P1d1[i]);
        
        //if( my_id < 14){  //13, k=70  (mp%nstep) == 0 &&
        
        /*if(my_id < nz_myid){

            for(int k = 0; k < NZ; k++){
                for(int i = 0; i < ni; i++){
                    int i1 = ib + i -1;
                    pT2d[k][i] = T[k][i1];
                    pP2d[k][i] = p00 * P[k][i1]; 
                }
            }
            MPI_File_seek(tmp_file1, sizeof(int), MPI_SEEK_CUR);
	        MPI_File_write(tmp_file1, &Istep, 1, MPI_INT, &status);             
            MPI_File_write(tmp_file1, &tt[i], 1, MPI_DOUBLE, &status);
            MPI_File_seek(tmp_file1, sizeof(int), MPI_SEEK_CUR);

            if(my_id == 0) printf("WRITE DMD data T2d ...\n");

            MPI_File_seek(tmp_file1, sizeof(int), MPI_SEEK_CUR);
            MPI_File_seek(tmp_file1, NP[my_id]*(num1*sizeof(double)), MPI_SEEK_CUR); 
            for(int k = 0; k < NZ; k++){
           
                MPI_File_write(tmp_file1, T2d+num1*k, num1,  MPI_DOUBLE, &status);
            
            }
    
            MPI_File_seek(tmp_file1, NP[nz_myid - 1 - my_id]*(num1*sizeof(double)), MPI_SEEK_CUR);  
            MPI_File_seek(tmp_file1, sizeof(int), MPI_SEEK_CUR);

            
            
            
            MPI_File_seek(tmp_file2, sizeof(int), MPI_SEEK_CUR);
	        MPI_File_write(tmp_file2, &Istep, 1, MPI_INT, &status);             
            MPI_File_write(tmp_file2, &tt[i], 1, MPI_DOUBLE, &status);
            MPI_File_seek(tmp_file2, sizeof(int), MPI_SEEK_CUR);

            if(my_id == 0) printf("WRITE DMD data P2d ...\np00 = %lf\n",p00);

            MPI_File_seek(tmp_file2, sizeof(int), MPI_SEEK_CUR);
            MPI_File_seek(tmp_file2, NP[my_id]*(num1*sizeof(double)), MPI_SEEK_CUR); 
            for(int k = 0; k < NZ; k++){
           
                MPI_File_write(tmp_file2, P2d+num1*k, num1,  MPI_DOUBLE, &status);
            
            }
    
            MPI_File_seek(tmp_file2, NP[nz_myid - 1 - my_id]*(num1*sizeof(double)), MPI_SEEK_CUR);  
            MPI_File_seek(tmp_file2, sizeof(int), MPI_SEEK_CUR);

        }*/

    }
    MPI_File_close(&tmp_file);
    /*MPI_File_close(&tmp_file1);
    MPI_File_close(&tmp_file2);
    if(my_id == 0) printf("READ Savedata OK \nWRITE DMD data OK\n");*/

    MPI_Barrier(MPI_COMM_WORLD);

    sprintf(filename, "test.dat");
    sprintf(filename1, "time1d-XZ%03d.dat",i0);
    //sprintf(filename2, "test-DMD.dat");
    
    if(my_id == 0){

        printf("write test data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,y,z,d,u,v,w,T,P,d0,u0,v0,w0,T0,P0\n");
        fprintf(fp, "zone i = %d, j = %d , k = %d\n", 1, nx, nz);
        fclose(fp); 

        printf("write test1d data ...\n");
        
        fp1 = fopen(filename1, "w");

        fprintf(fp1, "variables=tt,P1,P2,T1,T2,d1,d2,u1,u2\n");
        fprintf(fp1, "zone i=%d \n", mp);
        for(int i = 0; i < mp; i++){

            fprintf(fp1, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", tt[i], P1d0[i], P1d1[i], T1d0[i], T1d1[i], d1d0[i], d1d1[i], u1d0[i], u1d1[i]);
        }

        fclose(fp1);

        /*printf("write test-DMD data ...\n");
        
        fp2 = fopen(filename2, "w");
       
        fprintf(fp2, "variables=\n");
        fprintf(fp2, "zone i = %d, j = %d , k = %d\n", 1, nx, nz);
        fclose(fp2); */
    }
      

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < nx; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                            xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j],
                            ppw_buff[k][j], ppT_buff[k][j], ppP_buff[k][j], ppd0_buff[k][j], ppu0_buff[k][j], ppv0_buff[k][j], 
                            ppw0_buff[k][j], ppT0_buff[k][j], ppP0_buff[k][j]);
                        
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
                MPI_Send(pP_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pP0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pdrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(purms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pvrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pwrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pTrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pPrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
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
                MPI_Recv(pP_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pP0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pdrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(purms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pvrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pwrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pTrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pPrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }
        
}

void flow1d_tt(int m, int j, double U){   //不好用
    double (*d)[ny] = (double (*)[ny])(pd1);
    double (*u)[ny] = (double (*)[ny])(pu1);
    double (*v)[ny] = (double (*)[ny])(pv1);
    double (*w)[ny] = (double (*)[ny])(pw1);
    double (*T)[ny] = (double (*)[ny])(pT1);
    double (*P)[ny] = (double (*)[ny])(pP1);
    int  n, id;   //m为壁面法向网格编号
    //double U;
    double tmp;
    for(int i = 0; i < n_processe-1; i++){       //寻找m1,m2分别属于哪个my_id
        if(NP[i]-1 < m && NP[i+1]-1 > m){
            id = i;
        }                    
    }  
        n = m - NP[id];                   //寻找所属my_id内的壁面法向编号
    
        if(my_id == id){                    
            tmp =P[n][j];        //在对应my_id内赋值,j=180                        
        }

        MPI_Barrier(MPI_COMM_WORLD);         //阻断
        MPI_Bcast(&tmp, 1, MPI_DOUBLE, id, MPI_COMM_WORLD);   //将所赋值广播给所有节点
                
        if(my_id != id){
            U = tmp;         //在节点上赋值
            //printf("U = %15.6f\n",U);
        }
    
}


void Compute_frequancy_XZ(double U1d[], int i0, int i1, char* Uf){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
    
    double (*d)[nx] = (double (*)[nx])(pd1);
    double (*u)[nx] = (double (*)[nx])(pu1);
    double (*v)[nx] = (double (*)[nx])(pv1);
    double (*w)[nx] = (double (*)[nx])(pw1);
    double (*T)[nx] = (double (*)[nx])(pT1);
    double (*P)[nx] = (double (*)[nx])(pP1);

    double dt = tt[1] - tt[0];
    double omeg;
    double St;
    double favP0 = 0.0;
    double complex CI = 0.0 + 1.0 * I;       //复数
    //double U1d[mp];
    
    double x;
    double Uamplitude[1000];  //后面一组纬度等于无量纲频率omeg的数目
    
    
    char filename[120];
    
    if(my_id == 0){
        for(int i = 0; i < mp; i++){
            favP0 = favP0 +  U1d[i];    //所有时间的同一点处的物理量相加
        }

        favP0 = favP0 / mp;                //同一点处物理量的均值
        //printf("favP0 = %15.6f\n", favP0);

        for(int i = 0; i < mp; i++){
            U1d[i] = U1d[i] - favP0;    //同一点处所有时间物理量的脉动值
            //printf("U1d = %15.6f\n", U1d[i]);
        }                                                                             

        for(int i = 0; i < 1000; i++){
            omeg = i;                           //无量纲频率从0到999变化
            double complex UC = 0.0 + 0.0 * I ; //每次循环都将UC更新为0，防止UC在原有数值上累加
            for(int j = 0; j < mp; j++ ){ 
                x = j * deltx;
                UC = UC + U1d[j] * cexp(CI * omeg * x);      //做傅里叶变换得到幅值UC
            }
            Uamplitude[i] = 2.0 * cabs(UC) / mp;       //since abs(f(w))=abs(f(-w)), so the amplitude is two times !!!
            //printf("Uamplitude = %15.6f\n", Uamplitude[i]);
        }
    }

    sprintf(filename, "spectrumXZ-%03d-%d-%s.dat", i0, i1, Uf);   //读取不同截面的时候需要改

    if(my_id == 0){

        printf("write frequancy data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables= omeg*, omeg, St, %s \n", Uf);
        
        printf("dt0 = %15.6f\n", dt0);

        for(int i = 0; i < 1000; i++){
            
            omeg = i /(mp * dt0)/1000.0;

            St = omeg * 0.001/1069.1;   //St=fl/U , l=1mm, U=1069.1 m/s

            fprintf(fp, "%d%15.6f%15.6f%15.6f\n", i, omeg, St, Uamplitude[i]);
        }
        fclose(fp);
    }


}

void Compute_frequancy_2dXZ( int i0 ,int wk){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[nx] = (double (*)[nx])(pd1);
    double (*u)[nx] = (double (*)[nx])(pu1);
    double (*v)[nx] = (double (*)[nx])(pv1);
    double (*w)[nx] = (double (*)[nx])(pw1);
    double (*T)[nx] = (double (*)[nx])(pT1);
    double (*P)[nx] = (double (*)[nx])(pP1);

    double (*d0)[nx] = (double (*)[nx])(pd0);
    double (*u0)[nx] = (double (*)[nx])(pu0);
    double (*v0)[nx] = (double (*)[nx])(pv0);
    double (*w0)[nx] = (double (*)[nx])(pw0);
    double (*T0)[nx] = (double (*)[nx])(pT0);
    double (*P0)[nx] = (double (*)[nx])(pP0);

    double (*d2)[nx] = (double (*)[nx])(pd2);
    double (*u2)[nx] = (double (*)[nx])(pu2);
    double (*v2)[nx] = (double (*)[nx])(pv2);
    double (*w2)[nx] = (double (*)[nx])(pw2);
    double (*T2)[nx] = (double (*)[nx])(pT2);
    double (*P2)[nx] = (double (*)[nx])(pP2);

    double *x3d_buff, *y3d_buff, *z3d_buff;
    double *pd2_buff, *pu2_buff, *pv2_buff, *pw2_buff, *pT2_buff, *pP2_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd2_buff = (double*)malloc(nx * NZ * sizeof(double));
    pu2_buff = (double*)malloc(nx * NZ * sizeof(double));
    pv2_buff = (double*)malloc(nx * NZ * sizeof(double));
    pw2_buff = (double*)malloc(nx * NZ * sizeof(double));
    pT2_buff = (double*)malloc(nx * NZ * sizeof(double));
    pP2_buff = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd2_buff)[nx] = (double(*)[nx])pd2_buff;
    double (*ppu2_buff)[nx] = (double(*)[nx])pu2_buff;
    double (*ppv2_buff)[nx] = (double(*)[nx])pv2_buff;
    double (*ppw2_buff)[nx] = (double(*)[nx])pw2_buff;
    double (*ppT2_buff)[nx] = (double(*)[nx])pT2_buff;
    double (*ppP2_buff)[nx] = (double(*)[nx])pP2_buff;
   
    int num =  nx;
    int num_byte =  nx * sizeof(double);
    //int mp = 100;    //100个时间序列，可以按需求更改
    int Istep;
    //double tt[mp];  //读取mp个时间序列，tt存放无量纲时间
    char fp_name[120];
    char filename[120];
    char filename1[120];
    double tmpP1[mp];
    double tmpP2[mp];
    double tmpP0[mp];
    double tmp;

    MPI_File tmp_file;

    double complex CI = 0.0 + 1.0 * I;       //复数
    double complex Ctmp;
    

    sprintf(fp_name, "Savedata-XZ%03d.dat", i0);

    MPI_File_open(MPI_COMM_WORLD, fp_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    for(int i = 0; i < mp; i++ ){                      //读取mp个时间序列

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	    MPI_File_read_all(tmp_file, &Istep, 1, MPI_INT, &status);             
        MPI_File_read_all(tmp_file, &tt[i], 1, MPI_DOUBLE, &status);
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        if(my_id == 0)printf("Istep=%05d, tt=%lf\n", Istep, tt[i]);



        if(my_id == 0) printf("READ Savedata d ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pd1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata u ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pu1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata v ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pv1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata w ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pw1+num*k, num,  MPI_DOUBLE, &status);
          
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata T ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)  ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pT1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR); 
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < nx; j++){
                P[k][j]= d[k][j] * T[k][j];

                d[k][j] = d[k][j] - d0[k][j];
                u[k][j] = u[k][j] - u0[k][j];
                v[k][j] = v[k][j] - v0[k][j];
                w[k][j] = w[k][j] - w0[k][j];
                T[k][j] = T[k][j] - T0[k][j];
                P[k][j] = P[k][j] - P0[k][j];

                Ctmp = cexp(CI * wk * i * deltx);

                d2[k][j] = d2[k][j] + Ctmp * d[k][j];
                u2[k][j] = u2[k][j] + Ctmp * u[k][j];
                v2[k][j] = v2[k][j] + Ctmp * v[k][j];
                w2[k][j] = w2[k][j] + Ctmp * w[k][j];
                T2[k][j] = T2[k][j] + Ctmp * T[k][j];
                P2[k][j] = P2[k][j] + Ctmp * P[k][j];    


            }
        }

    }
    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("READ Savedata OK \n");

    MPI_Barrier(MPI_COMM_WORLD);

    sprintf(filename, "wallpressuredisturbance-%03d-%d.dat", i0,mp);  //输出沿流向的壁面压力脉动
    if(my_id == 0){

        printf("write wall pressure disturbance data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x, pressuredisturbance \n");
        for(int i = 0; i < nx; i++){
            fprintf(fp, "%15.6f%15.6f\n",xx3d[0][0][i], P[0][i] );
        }
        
        fclose(fp);
    }


    sprintf(filename, "spectrum2dxz-%03d-%d.dat", i0, wk);   //读取不同截面的时候需要改,i0表示读取第几个XZ截面，i1表示物理量及其对应的位置

    if(my_id == 0){

        printf("write 2d frequancy data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x, y, z, d2, u2, v2, w2, T2, P2 \n");
        fprintf(fp, "zone i = %d, j = %d , k = %d\n", 1, nx, nz);
        fclose(fp);
    }
    
        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < nx; j++){
                xx3d_buff[k][j] = xx3d[k][nypoints[i0]][j];
                yy3d_buff[k][j] = yy3d[k][nypoints[i0]][j];
                zz3d_buff[k][j] = zz3d[k][nypoints[i0]][j];
                ppd2_buff[k][j] = cabs(d2[k][j]);
                ppu2_buff[k][j] = cabs(u2[k][j]);
                ppv2_buff[k][j] = cabs(v2[k][j]);
                ppw2_buff[k][j] = cabs(w2[k][j]);
                ppT2_buff[k][j] = cabs(T2[k][j]);
                ppP2_buff[k][j] = cabs(P2[k][j]);
            }
        }

    get_shapefunction( np, nxpoints[0],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);
    get_shapefunction( np, nxpoints[1],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk); 
    get_shapefunction( np, nxpoints[2],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);
    get_shapefunction( np, nxpoints[3],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);
    get_shapefunction( np, nxpoints[4],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);
    get_shapefunction( np, nxpoints[5],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);
    get_shapefunction( np, nxpoints[6],ppd2_buff,ppu2_buff,ppv2_buff,ppw2_buff,ppT2_buff,ppP2_buff,wk);

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < nx; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                                xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j],
                                ppd2_buff[k][j], ppu2_buff[k][j], ppv2_buff[k][j], ppw2_buff[k][j], ppT2_buff[k][j], ppP2_buff[k][j]);
                    }
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pP2_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pP2_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }


}


void get_shapefunction(int i0,int xx,double Ud2[][nx],double Uu2[][nx],double Uv2[][nx],double Uw2[][nx],double UT2[][nx],double UP2[][nx],int wk){    //xx:流向位置坐标编号
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x3d_buff, *y3d_buff, *z3d_buff;

    x3d_buff = (double*)malloc( NZ * sizeof(double));
    y3d_buff = (double*)malloc( NZ * sizeof(double));
    z3d_buff = (double*)malloc( NZ * sizeof(double));

    double (*xx3d_buff) = (double(*))x3d_buff;
    double (*yy3d_buff) = (double(*))y3d_buff;
    double (*zz3d_buff) = (double(*))z3d_buff;

    double dsf[nz],usf[nz],vsf[nz],wsf[nz],Tsf[nz],Psf[nz];
    double rr[nz];
    double tmpx,tmpy,tmpz;
    char filename[120];

    if(my_id == 0){
        tmpx = xx3d[0][nypoints[i0]][xx];
        tmpy = yy3d[0][nypoints[i0]][xx];
        tmpz = zz3d[0][nypoints[i0]][xx];
    }
    //MPI_Barrier(MPI_COMM_WORLD);         //阻断
    MPI_Bcast(&tmpx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);   //将所赋值广播给所有节点
    MPI_Bcast(&tmpy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tmpz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);            
    if(my_id != 0){
        tmpx = tmpx;
        tmpy = tmpy;
        tmpz = tmpz;         //在节点上赋值
    }
    for(int k = 0; k < NZ; k++){
        xx3d_buff[k] = xx3d[k][nypoints[i0]][xx];
        yy3d_buff[k] = yy3d[k][nypoints[i0]][xx];
        zz3d_buff[k] = zz3d[k][nypoints[i0]][xx];
        rr[k] = sqrt( pow((xx3d_buff[k]-tmpx),2) + pow((yy3d_buff[k]-tmpy),2) + pow((zz3d_buff[k]-tmpz),2));
        dsf[k] = Ud2[k][xx];
        usf[k] = Uu2[k][xx];
        vsf[k] = Uv2[k][xx];
        wsf[k] = Uw2[k][xx];
        Tsf[k] = UT2[k][xx];
        Psf[k] = UP2[k][xx];          
    }

    sprintf(filename, "shapefunction-%03d-%d-%d.dat", i0, (int)(xx3d_buff[0]),wk);  //强制类型转换

    if(my_id == 0){

        printf("write shapefunction data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=rr, d, u, v, w, T, P \n");
        fprintf(fp, "zone i = %d\n", nz);
        fclose(fp);
    }
    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                                rr[k],dsf[k],usf[k], vsf[k], wsf[k], Tsf[k], Psf[k]);
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(dsf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(usf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(vsf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(wsf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(Tsf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(Psf, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(rr, NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(dsf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(usf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(vsf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(wsf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(Tsf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(Psf, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(rr, NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }

}

void Compute_Raa( int i0 ){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[nx] = (double (*)[nx])(pd1);
    double (*u)[nx] = (double (*)[nx])(pu1);
    double (*v)[nx] = (double (*)[nx])(pv1);
    double (*w)[nx] = (double (*)[nx])(pw1);
    double (*T)[nx] = (double (*)[nx])(pT1);
    double (*P)[nx] = (double (*)[nx])(pP1);

    double (*d0)[nx] = (double (*)[nx])(pd0);
    double (*u0)[nx] = (double (*)[nx])(pu0);
    double (*v0)[nx] = (double (*)[nx])(pv0);
    double (*w0)[nx] = (double (*)[nx])(pw0);
    double (*T0)[nx] = (double (*)[nx])(pT0);
    double (*P0)[nx] = (double (*)[nx])(pP0);

    double (*pdtmp)[NZ][nx] = (double (*)[NZ][nx])(dtmp);
    double (*putmp)[NZ][nx] = (double (*)[NZ][nx])(utmp);
    double (*pvtmp)[NZ][nx] = (double (*)[NZ][nx])(vtmp);
    double (*pwtmp)[NZ][nx] = (double (*)[NZ][nx])(wtmp);
    double (*pTtmp)[NZ][nx] = (double (*)[NZ][nx])(Ttmp);
    double (*pPtmp)[NZ][nx] = (double (*)[NZ][nx])(Ptmp);

    /*double (*pdtmpRaa)[nz][ny] = (double (*)[nz][ny])(dtmpRaa);
    double (*putmpRaa)[nz][ny] = (double (*)[nz][ny])(utmpRaa);
    double (*pvtmpRaa)[nz][ny] = (double (*)[nz][ny])(vtmpRaa);
    double (*pwtmpRaa)[nz][ny] = (double (*)[nz][ny])(wtmpRaa);
    double (*pTtmpRaa)[nz][ny] = (double (*)[nz][ny])(TtmpRaa);
    double (*pPtmpRaa)[nz][ny] = (double (*)[nz][ny])(PtmpRaa);*/

    double (*Rdd) = (double(*))malloc(nx*sizeof(double));
    double (*Ruu) = (double(*))malloc(nx*sizeof(double));
    double (*Rvv) = (double(*))malloc(nx*sizeof(double));
    double (*Rww) = (double(*))malloc(nx*sizeof(double));
    double (*RTT) = (double(*))malloc(nx*sizeof(double));
    double (*RPP) = (double(*))malloc(nx*sizeof(double));
   
    int num =  nx;
    int num_byte =  nx * sizeof(double);
    //int mp = 100;    //100个时间序列，可以按需求更改
    int Istep;
    //double tt[mp];  //读取mp个时间序列，tt存放无量纲时间
    char fp_name[120];
    char filename[120];
    char filename1[120];
    double tmpP1[mp];
    double tmpP2[mp];
    double tmpP0[mp];
    double tmp;

    MPI_File tmp_file;
    

    sprintf(fp_name, "Savedata-XZ%03d.dat", i0);

    MPI_File_open(MPI_COMM_WORLD, fp_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    for(int i = 0; i < mp; i++ ){                      //读取mp个时间序列

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	    MPI_File_read_all(tmp_file, &Istep, 1, MPI_INT, &status);             
        MPI_File_read_all(tmp_file, &tt[i], 1, MPI_DOUBLE, &status);
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        if(my_id == 0)printf("Istep=%05d, tt=%lf\n", Istep, tt[i]);



        if(my_id == 0) printf("READ Savedata d ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pd1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double)), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata u ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pu1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata v ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
            
            MPI_File_read(tmp_file, pv1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata w ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pw1+num*k, num,  MPI_DOUBLE, &status);
          
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        
        if(my_id == 0) printf("READ Savedata T ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)  ), MPI_SEEK_CUR);  
        for(int k = 0; k < NZ; k++){
           
            MPI_File_read(tmp_file, pT1+num*k, num,  MPI_DOUBLE, &status);
            
        }
    
        MPI_File_seek(tmp_file, NP[n_processe - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR); 
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

        
        
        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < nx; j++){
                P[k][j]= d[k][j] * T[k][j];

                pdtmp[i][k][j] = d[k][j] - d0[k][j];
                putmp[i][k][j] = u[k][j] - u0[k][j];
                pvtmp[i][k][j] = v[k][j] - v0[k][j];
                pwtmp[i][k][j] = w[k][j] - w0[k][j];
                pTtmp[i][k][j] = T[k][j] - T0[k][j];
                pPtmp[i][k][j] = P[k][j] - P0[k][j];
  
            }
        }

    }
    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("generate Raatmp OK \n");

    MPI_Barrier(MPI_COMM_WORLD);

    //get_data_yz(y2d,yy3d);
    //get_data_yz(z2d,zz3d);
    /*get_data_yz(pdtmpRaa,pdtmp);
    get_data_yz(putmpRaa,putmp);
    get_data_yz(pvtmpRaa,pvtmp);
    get_data_yz(pwtmpRaa,pwtmp);
    get_data_yz(pTtmpRaa,pTtmp);
    get_data_yz(pPtmpRaa,pPtmp);*/
    //MPI_Allgatherv( dtmp , mp*ny*NZ , MPI_DOUBLE , dtmp_Raa , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
    
    int kr0,id,n;

    for(int i = 0; i < n_processe; i++){       //寻找m1,m2分别属于哪个my_id
        if(NP[i] <= k0 && NPE[i] > k0){
            id = i;
        }                    
    }  
        n = k0 - NP[id];                   //寻找所属my_id内的壁面法向编号
    
    printf("id = %d,n = %d \n",id,n);

    sprintf(filename, "RaaXZ-%05d.dat", nypoints[i0] );

    if(my_id == id){
    printf("1\n");
    for(int kr=0; kr<nx-641+1; kr++){             //扰动在90-100之间施加
        for(int k=641; k<nx; k++){
            for(int m=0; m<mp; m++){
                if(k+kr > nx){
                    kr0 = kr - nx+1+641;
                }else{
                    kr0 = kr;
                }
                Rdd[kr] = Rdd[kr] + pdtmp[m][n][k] * pdtmp[m][n][k+kr0];
                Ruu[kr] = Ruu[kr] + putmp[m][n][k] * putmp[m][n][k+kr0];
                Rvv[kr] = Rvv[kr] + pvtmp[m][n][k] * pvtmp[m][n][k+kr0];
                Rww[kr] = Rww[kr] + pwtmp[m][n][k] * pwtmp[m][n][k+kr0];
                RTT[kr] = RTT[kr] + pTtmp[m][n][k] * pTtmp[m][n][k+kr0];
                RPP[kr] = RPP[kr] + pPtmp[m][n][k] * pPtmp[m][n][k+kr0];
            }
        }
        Rdd[kr] = Rdd[kr]/mp;
        Ruu[kr] = Ruu[kr]/mp;
        Rvv[kr] = Rvv[kr]/mp;
        Rww[kr] = Rww[kr]/mp;
        RTT[kr] = RTT[kr]/mp;
        RPP[kr] = RPP[kr]/mp;
        printf("kr = %d\n", kr);
    }

    
    printf("write RaaYZ data ...\n");
        
        fp = fopen(filename, "w");
       
        fprintf(fp, "variables=x,Rdd,Ruu,Rvv,Rww,RTT,Rpp \n");
        
        for(int kr = 0; kr<(int)((nx-641+1)/2); kr++){
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                    xx3d[0][0][kr+641],Rdd[kr]/Rdd[0],Ruu[kr]/Ruu[0],Rvv[kr]/Rvv[0],Rww[kr]/Rww[0],RTT[kr]/RTT[0],RPP[kr]/RPP[0]);

        }


        fclose(fp);
    }



}

void Write_Savedata_timeaverage_XZ(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d0)[nx] = (double (*)[nx])(pd0);
    double (*u0)[nx] = (double (*)[nx])(pu0);
    double (*v0)[nx] = (double (*)[nx])(pv0);
    double (*w0)[nx] = (double (*)[nx])(pw0);
    double (*T0)[nx] = (double (*)[nx])(pT0);
    double (*P0)[nx] = (double (*)[nx])(pP0);

    double (*drms)[nx] = (double (*)[nx])(pdrms);
    double (*urms)[nx] = (double (*)[nx])(purms);
    double (*vrms)[nx] = (double (*)[nx])(pvrms);
    double (*wrms)[nx] = (double (*)[nx])(pwrms);
    double (*Trms)[nx] = (double (*)[nx])(pTrms);
    double (*Prms)[nx] = (double (*)[nx])(pPrms);

    double *x3d_buff, *y3d_buff, *z3d_buff;
    double *pd0_buff, *pu0_buff, *pv0_buff, *pw0_buff, *pT0_buff, *pP0_buff;
    double *pdrms_buff, *purms_buff, *pvrms_buff, *pwrms_buff, *pTrms_buff, *pPrms_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pP0_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pdrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    purms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pvrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pwrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pTrms_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pPrms_buff  = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd0_buff)[nx]  = (double(*)[nx])pd0_buff;
    double (*ppu0_buff)[nx]  = (double(*)[nx])pu0_buff;
    double (*ppv0_buff)[nx]  = (double(*)[nx])pv0_buff;
    double (*ppw0_buff)[nx]  = (double(*)[nx])pw0_buff;
    double (*ppT0_buff)[nx]  = (double(*)[nx])pT0_buff;
    double (*ppP0_buff)[nx]  = (double(*)[nx])pP0_buff;
    double (*ppdrms_buff)[nx]  = (double(*)[nx])pdrms_buff;
    double (*ppurms_buff)[nx]  = (double(*)[nx])purms_buff;
    double (*ppvrms_buff)[nx]  = (double(*)[nx])pvrms_buff;
    double (*ppwrms_buff)[nx]  = (double(*)[nx])pwrms_buff;
    double (*ppTrms_buff)[nx]  = (double(*)[nx])pTrms_buff;
    double (*ppPrms_buff)[nx]  = (double(*)[nx])pPrms_buff;

    double tmp = 1.0 / mp;
    char filename[120];
    char filename1[120];

    //printf("tmp = %15.6f\n", tmp);

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < nx; j++){
            
            d0[k][j] = d0[k][j] * tmp;
            u0[k][j] = u0[k][j] * tmp;
            v0[k][j] = v0[k][j] * tmp;
            w0[k][j] = w0[k][j] * tmp;          //mp个截面进行平均                   
            T0[k][j] = T0[k][j] * tmp;
            P0[k][j] = P0[k][j] * tmp;

            drms[k][j] = drms[k][j] * tmp;
            urms[k][j] = urms[k][j] * tmp;
            vrms[k][j] = vrms[k][j] * tmp;          //mp个截面进行平均 
            wrms[k][j] = wrms[k][j] * tmp;
            Trms[k][j] = Trms[k][j] * tmp;
            Prms[k][j] = Prms[k][j] * tmp;

            drms[k][j] = sqrt( fabs(drms[k][j] - d0[k][j] * d0[k][j] ) );     
            urms[k][j] = sqrt( fabs(urms[k][j] - u0[k][j] * u0[k][j] ) );
            vrms[k][j] = sqrt( fabs(vrms[k][j] - v0[k][j] * v0[k][j] ) );
            wrms[k][j] = sqrt( fabs(wrms[k][j] - w0[k][j] * w0[k][j] ) );
            Trms[k][j] = sqrt( fabs(Trms[k][j] - T0[k][j] * T0[k][j] ) );
            Prms[k][j] = sqrt( fabs(Prms[k][j] - P0[k][j] * P0[k][j] ) );

            xx3d_buff[k][j] = xx3d[k][nypoints[i0]][j];      //设置缓冲数组防止数据交换时覆盖掉
            yy3d_buff[k][j] = yy3d[k][nypoints[i0]][j];
            zz3d_buff[k][j] = zz3d[k][nypoints[i0]][j];
            ppd0_buff[k][j]  = d0[k][j];
            ppu0_buff[k][j]  = u0[k][j];
            ppv0_buff[k][j]  = v0[k][j];
            ppw0_buff[k][j]  = w0[k][j];
            ppT0_buff[k][j]  = T0[k][j];
            ppP0_buff[k][j]  = P0[k][j];
            ppdrms_buff[k][j]  = drms[k][j];
            ppurms_buff[k][j]  = urms[k][j];
            ppvrms_buff[k][j]  = vrms[k][j];
            ppwrms_buff[k][j]  = wrms[k][j];
            ppTrms_buff[k][j]  = Trms[k][j];
            ppPrms_buff[k][j]  = Prms[k][j];

        }
    }           

    MPI_Barrier(MPI_COMM_WORLD);
    sprintf(filename, "XZplane-time-average-%03d-%d.dat",i0,mp);
    
    if(my_id == 0){

        printf("write time average data ...\n");

        fp = fopen(filename, "w");
        fprintf(fp, "variables=x,y,z,d0,u0,v0,w0,T0,P0,drms,urms,vrms,wrms,Trms,Prms\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, nx, nz);
        fclose(fp); 

    }  

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < nx; j++){
                        
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", 
                                xx3d_buff[k][j], yy3d_buff[k][j], zz3d_buff[k][j], 
                                ppd0_buff[k][j], ppu0_buff[k][j], ppv0_buff[k][j], ppw0_buff[k][j], ppT0_buff[k][j], ppP0_buff[k][j],
                                ppdrms_buff[k][j], ppurms_buff[k][j], ppvrms_buff[k][j], ppwrms_buff[k][j], ppTrms_buff[k][j], ppPrms_buff[k][j]);

                    }
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pP0_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pdrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(purms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pvrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pwrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pTrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pPrms_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pP0_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pdrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(purms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pvrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pwrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pTrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pPrms_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
    }

}

void DMD_XZ_coordinate(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x2d_buff, *y2d_buff, *z2d_buff;

    x2d_buff = (double*)malloc(ni * NZ * sizeof(double));
    y2d_buff = (double*)malloc(ni * NZ * sizeof(double));
    z2d_buff = (double*)malloc(ni * NZ * sizeof(double));

    double (*xx2d_buff)[ni] = (double(*)[ni])x2d_buff;
    double (*yy2d_buff)[ni] = (double(*)[ni])y2d_buff;
    double (*zz2d_buff)[ni] = (double(*)[ni])z2d_buff;

    int num =  ni;
    int num_byte =  ni * sizeof(double);
    
    char fp_name[120];
    char filename[120];

    MPI_File tmp_file;



    for(int k = 0; k < NZ; k++){
        for(int i = 0; i < ni; i++){
            
            int i1 = ib + i - 1;

            xx2d_buff[k][i] = xx3d[k][nypoints[i0]][i1];      //设置缓冲数组防止数据交换时覆盖掉
            //yy3d_buff[k][i] = yy3d[k][nypoints[i0]][i1];
            zz2d_buff[k][i] = zz3d[k][nypoints[i0]][i1];

        }
    }

    sprintf(fp_name, "DMD-coordinate.dat");

    MPI_File_open(MPI_COMM_WORLD, fp_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id < nz_myid){
    
    //for(int i = 0; i < mp; i++ ){                      //mp个时间序列

        /*MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	    MPI_File_read_all(tmp_file, &Istep, 1, MPI_INT, &status);             
        MPI_File_read_all(tmp_file, &tt[i], 1, MPI_DOUBLE, &status);
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);*/
    
        if(my_id == 0) printf("WRITE DMD coordinate data x ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
        for(int k = 0; k < NZ; k++){
           
            MPI_File_write(tmp_file, x2d_buff+num*k, num,  MPI_DOUBLE, &status);
            
        }
        MPI_File_seek(tmp_file, NP[nz_myid - 1 - my_id]*(num*sizeof(double) ), MPI_SEEK_CUR);  
        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

        if(my_id == 0) printf("WRITE DMD coordinate data z ...\n");

        MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
        MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double)), MPI_SEEK_CUR); 
        for(int k = 0; k < NZ; k++){
           
            MPI_File_write(tmp_file, z2d_buff+num*k, num,  MPI_DOUBLE, &status);
            
        }

    }

    MPI_File_close(&tmp_file);
    if(my_id == 0) printf("WRITE DMD coordinate data OK \n");

    MPI_Barrier(MPI_COMM_WORLD);

    sprintf(filename, "DMD-coordinate-tec.dat");

    if(my_id == 0){

        printf("WRITE DMD coordinate tec data...\n");

        fp = fopen(filename, "w");
        fprintf(fp, "variables=x,z\n");
        fprintf(fp, "zone i=%d ,j=%d \n",  ni, nz);
        fclose(fp); 

    } 

    for(int n = 0; n < n_processe; n++){
            
            if(my_id == 0){

                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ni; j++){
                        
                            fprintf(fp, "%15.6f%15.6f\n", 
                                xx2d_buff[k][j], zz2d_buff[k][j]);  //, yy2d_buff[k][j]

                    }
                }

                fclose(fp);
            }    
    
            if(my_id != 0){
                MPI_Send(x2d_buff, ni*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y2d_buff, ni*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z2d_buff, ni*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                
            }

            if(my_id != n_processe-1){
                MPI_Recv(x2d_buff, ni*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y2d_buff, ni*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z2d_buff, ni*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                
            }
    }




}


void Read_data(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "opencfd.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file); //MPI_File_open(MPI_COMM_WORLD, "opencfd.ana", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_read_all(tmp_file, init, 1, MPI_INT, &status);             
    MPI_File_read_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pd+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ u ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pu+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ v ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pv+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ w ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pw+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ T ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pT+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
    printf("READ data OK\n");
}


void Write_data_new_extension(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    int DI = sizeof(int) + sizeof(double);
    MPI_File tmp_file;
    double *tmp2d;
    int nx_new = nx - 25;
    int num_new = nx_new*ny;

    tmp2d = (double*)malloc(nx*ny * sizeof(double));

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double (*tmp)[nx] = (double (*)[nx])(tmp2d);


    MPI_File_open(MPI_COMM_WORLD, "opencfd.md", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);
	MPI_File_write_all(tmp_file, &init, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &init[1], 1, MPI_DOUBLE, &status);
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);

    if(my_id == 0) printf("WRITE d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = d[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = d[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE u ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = u[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = u[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE v ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = v[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = v[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE w ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = w[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = w[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE T ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = T[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = T[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}


void Write_data_new(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    int DI = sizeof(int) + sizeof(double);
    MPI_File tmp_file;
    double *tmp2d;
    int nx_new = nx - 10;
    int num_new = nx_new*ny;

    tmp2d = (double*)malloc(num_new * sizeof(double));

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double (*tmp)[nx_new] = (double (*)[nx_new])(tmp2d);


    MPI_File_open(MPI_COMM_WORLD, "opencfd.md", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);
	MPI_File_write_all(tmp_file, &init, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &init[1], 1, MPI_DOUBLE, &status);
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);

    if(my_id == 0) printf("WRITE d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = d[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE u ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = u[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE v ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = v[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE w ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = w[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE T ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = T[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}

void Write_grid_format(){
    int div = 2;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    if(my_id == 0) printf("Write grid.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc(div * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "grid%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], ny, nz);
            fclose(fp);
        }
    }

    for(int n = 0; n < n_processe; n++){
        for(int m = 0; m < div; m++){
            sprintf(filename, "grid%02d.dat", m);
    
            if(my_id == 0){
                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        for(int i = 0; i < MP[m]; i++){
                            int tmp = MP_offset[m] + i;
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[k][j][tmp], yy3d[k][j][tmp], zz3d[k][j][tmp]);
                        }
                    }
                }

                fclose(fp);
            }
        }

        if(my_id != 0){
            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void Write_data_format(){
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    if(my_id == 0) printf("Write grid.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc(div * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "grid%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], ny, nz);
            fclose(fp);
        }
    }

    for(int n = 0; n < n_processe; n++){
        for(int m = 0; m < div; m++){
            sprintf(filename, "grid%02d.dat", m);
    
            if(my_id == 0){
                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        for(int i = 0; i < MP[m]; i++){
                            int tmp = MP_offset[m] + i;
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[k][j][tmp], yy3d[k][j][tmp], zz3d[k][j][tmp], 
                            d[k][j][tmp], u[k][j][tmp], v[k][j][tmp], w[k][j][tmp], T[k][j][tmp]);
                        }
                    }
                }

                fclose(fp);
            }
        }

        if(my_id != 0){
            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);

            MPI_Send(pd, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);

            MPI_Recv(pd, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void Write_dataxy2d_first_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double cf, Tk;

    if(my_id == 0){
        printf("Write dataxy2d.dat\n");

        fp = fopen("dataxy.dat", "w");
        fprintf(fp, "variables=x,y,z,d,u,v,w,T,cf,Tk\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 2*ny-1, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][0][i] - T[0][0][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }
        for(int j = ny-2; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][ny-1][i] - T[0][ny-1][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                -yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }

        fclose(fp);
    }
}


void Write_dataxy2d_first_format_total(){                 //输出xy截面的信息，可以指定生成壁面上的信息
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double cf, Tk, hh;

    if(my_id == 0){
        printf("Write datawall.dat\n");

        fp = fopen("datawall.dat", "w");
        fprintf(fp, "variables=x,y,z,d,u,v,w,T,cf,Tk\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][0][i] - T[0][0][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][j][i],
                yy3d[0][j][i], zz3d[0][j][i], d[0][j][i], u[0][j][i], v[0][j][i], w[0][j][i], T[0][j][i], cf, Tk);
            }
        }

        fclose(fp);
    }
}


void Write_cf2d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk, hh;
    int m=391;                  //选择j=m截面作为输出壁面摩阻的截面

    if(my_id == 0){
        printf("Write cf2d.dat\n");

        fp = fopen("cf2d_bottom.dat", "w");        //推测是迎风面或背风面的一条摩阻曲线，非均匀网格下j要修改
        fprintf(fp, "variables=x,cf,Tk\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int i = 0; i < nx; i++){
            hh = sqrt((xx3d[1][m][i] - xx3d[0][m][i])*(xx3d[1][m][i] - xx3d[0][m][i]) + 
                      (yy3d[1][m][i] - yy3d[0][m][i])*(yy3d[1][m][i] - yy3d[0][m][i]) + 
                      (zz3d[1][m][i] - zz3d[0][m][i])*(zz3d[1][m][i] - zz3d[0][m][i]));

            cf = 2*Amu*sqrt(u[1][m][i]*u[1][m][i] + v[1][m][i]*v[1][m][i] + w[1][m][i]*w[1][m][i])/hh;

            Tk = 2*Amu*Cp/Pr*(T[1][m][i] - T[0][m][i])/hh;

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[0][m][i], cf, Tk, hh);
        }

        fclose(fp);

        /*fp = fopen("cf2d_top.dat", "w");          //非均匀网格下j要修改
        fprintf(fp, "variables=x,cf,Tk\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int i = 0; i < nx; i++){
            hh = sqrt((xx3d[1][ny-1][i] - xx3d[0][ny-1][i])*(xx3d[1][ny-1][i] - xx3d[0][ny-1][i]) + 
                      (yy3d[1][ny-1][i] - yy3d[0][ny-1][i])*(yy3d[1][ny-1][i] - yy3d[0][ny-1][i]) + 
                      (zz3d[1][ny-1][i] - zz3d[0][ny-1][i])*(zz3d[1][ny-1][i] - zz3d[0][ny-1][i]));

            cf = 2*Amu*sqrt(u[1][ny-1][i]*u[1][ny-1][i] + v[1][ny-1][i]*v[1][ny-1][i] + w[1][ny-1][i]*w[1][ny-1][i])/hh;

            Tk = 2*Amu*Cp/Pr*(T[1][ny-1][i] - T[0][ny-1][i])/hh;

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[1][ny-1][i], cf, Tk);
        }

        fclose(fp);*/
    }
}

void Write_cf3d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf;

    if(my_id == 0){
        printf("Write cf3d.dat\n");

        fp = fopen("cf3d.dat", "w");
        fprintf(fp, "variables=x,y,z,cf\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 2*ny-1, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], cf);
            }
        }
        for(int j = ny-2; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));
                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                -yy3d[1][j][i], zz3d[1][j][i], cf);
            }
        }

        fclose(fp);
    }
}


void Write_cf3d_format_total(){                                //输出壁面摩阻
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf;

    if(my_id == 0){
        printf("Write cf3d.dat\n");

        fp = fopen("cf3d.dat", "w");
        fprintf(fp, "variables=x,y,z,cf\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][j][i],
                yy3d[0][j][i], zz3d[0][j][i], cf);
            }
        }

        fclose(fp);
    }
}

void Write_datayz2d1_format(){//写出一个yz截面的数据，7/10位置
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny] = (double(*)[ny])x3d_buff;
    double (*yy3d_buff)[ny] = (double(*)[ny])y3d_buff;
    double (*zz3d_buff)[ny] = (double(*)[ny])z3d_buff;
    double (*ppd_buff)[ny]  = (double(*)[ny])pd_buff;
    double (*ppu_buff)[ny]  = (double(*)[ny])pu_buff;
    double (*ppv_buff)[ny]  = (double(*)[ny])pv_buff;
    double (*ppw_buff)[ny]  = (double(*)[ny])pw_buff;
    double (*ppT_buff)[ny]  = (double(*)[ny])pT_buff;


    if(my_id == 0) printf("Write datayz2d_1.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;

    int m=7;
    int tmp = MP_offset[m+1];

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            xx3d_buff[k][j] = xx3d[k][j][tmp];
            yy3d_buff[k][j] = yy3d[k][j][tmp];
            zz3d_buff[k][j] = zz3d[k][j][tmp];
            ppd_buff[k][j]  = d[k][j][tmp];
            ppu_buff[k][j]  = u[k][j][tmp];
            ppv_buff[k][j]  = v[k][j][tmp];
            ppw_buff[k][j]  = w[k][j][tmp];
            ppT_buff[k][j]  = T[k][j][tmp];
        }
    }

    if(my_id == 0){
        sprintf(filename, "datayz%02d.dat", m);
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, 2*ny-1, nz);
        fclose(fp);
    }


    for(int n = 0; n < n_processe; n++){

        sprintf(filename, "datayz%02d.dat", m);

        if(my_id == 0){
            fp = fopen(filename, "a");
    
            for(int k = 0; k < NPZ[n]; k++){

                for(int j = 0; j < ny; j++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                        yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                        ppw_buff[k][j], ppT_buff[k][j]);
                }
                for(int j = ny-2; j >= 0; j--){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                        -yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                        ppw_buff[k][j], ppT_buff[k][j]);
                }
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}

void Write_datayz2d_format(){
    int div = 2;          //把流场沿流向均匀取10个站位
    int n=1;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny] = (double(*)[ny])x3d_buff;
    double (*yy3d_buff)[ny] = (double(*)[ny])y3d_buff;
    double (*zz3d_buff)[ny] = (double(*)[ny])z3d_buff;
    double (*ppd_buff)[ny]  = (double(*)[ny])pd_buff;
    double (*ppu_buff)[ny]  = (double(*)[ny])pu_buff;
    double (*ppv_buff)[ny]  = (double(*)[ny])pv_buff;
    double (*ppw_buff)[ny]  = (double(*)[ny])pw_buff;
    double (*ppT_buff)[ny]  = (double(*)[ny])pT_buff;


    if(my_id == 0) printf("Write datayz2d.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;         //最后一个站位就是锥身末端

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "datayz%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, ny, nz);
//            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, 2*ny-1, nz);
            fclose(fp);
        }
    }


    for(int m = 0; m < div; m++){
        int tmp = MP_offset[m+1];

        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < ny; j++){
                xx3d_buff[k][j] = xx3d[k][j][tmp];
                yy3d_buff[k][j] = yy3d[k][j][tmp];
                zz3d_buff[k][j] = zz3d[k][j][tmp];
                ppd_buff[k][j]  = d[k][j][tmp];
                ppu_buff[k][j]  = u[k][j][tmp];
                ppv_buff[k][j]  = v[k][j][tmp];
                ppw_buff[k][j]  = w[k][j][tmp];
                ppT_buff[k][j]  = T[k][j][tmp];
            }
        }
        for(int n = 0; n < n_processe; n++){

            sprintf(filename, "datayz%02d.dat", m);

            if(my_id == 0){
                fp = fopen(filename, "a");
    
                for(int k = 0; k < NPZ[n]; k++){

                    for(int j = 0; j < ny; j++){
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                            yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                            ppw_buff[k][j], ppT_buff[k][j]);
                    }
//                    for(int j = ny-2; j >= 0; j--){
//                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
//                            -yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
//                            ppw_buff[k][j], ppT_buff[k][j]);
//                    }
                }

                fclose(fp);
            }


            if(my_id != 0){
                MPI_Send(x3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
}


void Write_dataxz2d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT_buff  = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;      //强制类型转换？
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd_buff)[nx]  = (double(*)[nx])pd_buff;
    double (*ppu_buff)[nx]  = (double(*)[nx])pu_buff;
    double (*ppv_buff)[nx]  = (double(*)[nx])pv_buff;
    double (*ppw_buff)[nx]  = (double(*)[nx])pw_buff;
    double (*ppT_buff)[nx]  = (double(*)[nx])pT_buff;

    int m=391;   //周向坐标j0=307

    if(my_id == 0) printf("Write dataxz2d.dat\n");
    char filename[100];

    for(int k = 0; k < NZ; k++){
        for(int i = 0; i < nx; i++){
            xx3d_buff[k][i] = xx3d[k][m][i];         //赋值为j=m截面
            yy3d_buff[k][i] = yy3d[k][m][i];
            zz3d_buff[k][i] = zz3d[k][m][i];
            ppd_buff[k][i]  = d[k][m][i];
            ppu_buff[k][i]  = u[k][m][i];
            ppv_buff[k][i]  = v[k][m][i];
            ppw_buff[k][i]  = w[k][m][i];
            ppT_buff[k][i]  = T[k][m][i];
        }
    }

//============================================输出流场信息====================================================
    if(my_id == 0){
        sprintf(filename, "dataxz.dat");
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);
   
        sprintf(filename, "flow2d-j0-LST.dat");       //为LST分析输出流场信息flow2d-j0-LST.dat
        fp1 = fopen(filename, "w");
   
        fprintf(fp1, "variables=x,y,d,u,v,w,T\n");
        fprintf(fp1, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);

        sprintf(filename, "grid.dat");                //为LST分析输出坐标信息grid.dat
        fp2 = fopen(filename, "w");

        fprintf(fp2, "variables=x,y,z\n");
        fprintf(fp2, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int i = 0; i < nx; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i], zz3d_buff[k][i], ppd_buff[k][i], ppu_buff[k][i], ppv_buff[k][i], 
                        ppw_buff[k][i], ppT_buff[k][i]);
                
                        fprintf(fp1, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i],  ppd_buff[k][i], ppu_buff[k][i], ppv_buff[k][i], 
                        ppw_buff[k][i], ppT_buff[k][i]);
                
                        fprintf(fp2, "%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i], zz3d_buff[k][i]);
                }

            }
        }
    
//=========================================为LST分析输出流场信息flow2d-j0-LST.dat====================================
    /*if(my_id == 0){
        sprintf(filename, "flow2d-j0-LST.dat");
        fp = fopen(filename, "w");

        //fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        //fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int i = 0; i < nx; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i],  ppd_buff[k][i], ppu_buff[k][i], ppv_buff[k][i], 
                        ppw_buff[k][i], ppT_buff[k][i]);
                }

            }
        }
    }
//==========================================为LST分析输出坐标信息grid.dat=====================================
    if(my_id == 0){
        sprintf(filename, "grid.dat");
        fp = fopen(filename, "w");

        //fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        //fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int i = 0; i < nx; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i], zz3d_buff[k][i]);
                }

            }
        }
    }*/
//============================================================================================================
        if(my_id != 0){
            MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);        
            MPI_Send(pu_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);     
            MPI_Recv(pu_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0){
        fclose(fp);
        fclose(fp1);
        fclose(fp2);
    } 
}


void Write_dataxy2d1_format(){                                
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);


    if(my_id == 0) printf("Write dataxy2d_1.dat\n");
    char filename[100];

    int m=(nz-1)/2, n, id = n_processe-1;

    for(int i = 0; i < n_processe-1; i++){
        if(NP[i] < m && NP[i+1] > m){
            id = i;
        }
    }

    n = m - NP[id];

    if(my_id == id){
        sprintf(filename, "dataxy%06d.dat", m);
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

    
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[n][j][i],
                    yy3d[n][j][i], zz3d[n][j][i], d[n][j][i], u[n][j][i], v[n][j][i], w[n][j][i], 
                    T[n][j][i]);
            }
        }

        fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


void average_data_xy(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*ny*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            data2d[j][i] = 0.0;
            data_buff[j][i] = 0.0;
            for(int k=0; k<NZ; k++){
                data_buff[j][i] += data3d[k][j][i];
            }
        }
    }

    MPI_Reduce(&data_buff[0][0], &data2d[0][0], nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            data2d[j][i] /= nz;
        }
    }
}


void average_data_xz(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            data_buff[k][i] = 0.0;
            for(int j=0; j<ny; j++){
                data_buff[k][i] += data3d[k][j][i];
            }
            data_buff[k][i] /= ny;                      //周向平均
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    data2d[NP[n] + k][i] = data_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pdata_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pdata_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void comput_yh_us_compress_ramp(double (*yh2d)[nx], double (*us2d)[nx], double (*u2d)[nx], double (*v2d)[nx]){//针对压缩折角，计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double length, length_min = 10000;
    double seta0 = 34.0*PI/180.0;
    int start_point = 8000;
    double seta;

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            us2d[j][i] = 0.0;
            //length_min = yy3d[0][j][i];
            length_min = 10000;
            //if(i >= start_point){
                for(int iw=0; iw<nx; iw++){
                    length = sqrt(pow(xx3d[0][j][iw] - xx3d[0][0][iw], 2) + 
                    pow(yy3d[0][j][iw] - yy3d[0][0][iw], 2));

                    if(length <= length_min) length_min = length;//计算格点与壁面最近的距离
                }
            //}
            yh2d[j][i] = length_min;

            if(xx3d[0][0][i] <= 0){
                seta = 0.0;
            }else{
                seta = seta0;/* 如果是在角区，则将速度投影至避面平行方向*/
            }

            us2d[j][i] = u2d[j][i]*cos(seta) + v2d[j][i]*sin(seta);
        }
    }
}


void comput_zh_us_compress_ramp(double (*zh2d)[nx], double (*us2d)[nx]){//针对压缩折角，计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);

    double length;
    double seta0 = 7.0*PI/180.0;
    double seta1 = 0.0;
    int start_point = 8000;
    double seta;
    //double *seta = (double*)malloc(ny*sizeof(double));

    double *pxx_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*xx_buff)[nx] = (double (*)[nx])(pxx_buff);

    double *pzz_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*zz_buff)[nx] = (double (*)[nx])(pzz_buff);

    double *puu_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*uu_buff)[nx] = (double (*)[nx])(puu_buff);

    double (*xx2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*zz2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));

    //double *pseta = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);
    //double (*seta)[ny][nx] = (double (*)[ny][nx])(pseta);
    
    //if(my_id == 0){       
    //printf("Write testseta1d.dat\n");
    //fp = fopen("testseta1d.dat", "w");
    /*for(int j=0; j<ny; j++){

                seta[j]=acos(zz3d[0][j][0]/sqrt(zz3d[0][j][0]*zz3d[0][j][0]+yy3d[0][j][0]*yy3d[0][j][0]));
            
                if(yy3d[0][j][0] < 0){
                seta[j]=2*PI-seta[j];
                
                //fprintf(fp, "%d%15.6f\n", j, seta[j]*180/PI);
                }
            //close(fp);
            } */           
    //}

    for(int i=0; i<nx; i++){
        for(int k=0; k<NZ; k++){
            for(int j=0; j<ny; j++){

                seta=acos(zz3d[0][j][0]/sqrt(zz3d[0][j][0]*zz3d[0][j][0]+yy3d[0][j][0]*yy3d[0][j][0]));
            
                if(yy3d[0][j][0] < 0){
                seta=2*PI-seta;
                }

                Vr[k][j][i] = v[k][j][i]*sin(seta) + w[k][j][i]*cos(seta);     //径向速度，对应周向均匀网格，非均匀网格需要改，读seta文件？
                Vseta[k][j][i] = v[k][j][i]*cos(seta) - w[k][j][i]*sin(seta);  //周向速度

                Vx[k][j][i] = u[k][j][i]*cos(seta0) + Vr[k][j][i]*sin(seta0);       //平行壁面速度
                Vr[k][j][i] = -u[k][j][i]*sin(seta0) + Vr[k][j][i]*cos(seta0);      //垂直于壁面的速度
            }
        }
    }

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            xx_buff[k][i] = xx3d[k][0][i];
            zz_buff[k][i] = zz3d[k][0][i];

            uu_buff[k][i] = 0.0;

            for (int j=0; j<ny; j++)       //非均匀网格需要改
            {
                uu_buff[k][i] += Vx[k][j][i];
            }

            uu_buff[k][i] /= ny;           //周向平均
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    xx2d[NP[n] + k][i] = xx_buff[k][i];
                    zz2d[NP[n] + k][i] = zz_buff[k][i];
                    us2d[NP[n] + k][i] = uu_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pxx_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pzz_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(puu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pxx_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pzz_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(puu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    for(int k=0; k<nz; k++){
        for(int i=0; i<nx; i++){

            length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + 
            pow(zz2d[k][i] - zz2d[0][i], 2));

            zh2d[k][i] = length;
        }
    }
}

void Write_OCFDYZ_Mesh(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x2d = (double*)malloc(ny * NZ * sizeof(double));
    double *y2d = (double*)malloc(ny * NZ * sizeof(double));
    double *z2d = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx2d)[ny] = (double (*)[ny])(x2d);
    double (*yy2d)[ny] = (double (*)[ny])(y2d);
    double (*zz2d)[ny] = (double (*)[ny])(z2d);

    //double seta;
    double *seta = (double*)malloc(ny*sizeof(double));

    if(my_id == 0) 
    {
        printf("Write xx1d.dat\n");
        fp = fopen("xx1d.dat", "w");
        for(int i = 0; i < nx; i++){
            fprintf(fp, "%d%15.6f\n", i, xx3d[0][0][i]);
        }
        fclose(fp);
    
        printf("Write seta1d.dat\n");
        fp = fopen("seta1d.dat", "w");
        for(int j = 0; j < ny; j++){
            seta[j]=acos(zz3d[0][j][0]/sqrt(zz3d[0][j][0]*zz3d[0][j][0]+yy3d[0][j][0]*yy3d[0][j][0]));
            
            if(yy3d[0][j][0] < 0){
                seta[j]=2*PI-seta[j];
            }
            
            fprintf(fp, "%d%15.6f\n", j, seta[j]*180/PI);
        }
        fclose(fp);
    
    
    }

   /* if(my_id == 0){
        printf("Write zz1d.dat\n");
        fp = fopen("zz1d.dat", "w");
    }
    
    for(int n = 0; n < n_processe; n++){
        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                fprintf(fp, "%d%15.6f\n", i, zz3d[k][504][i0]);
            }
        }*/

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            xx2d[k][j] = xx3d[k][j][i0];
            yy2d[k][j] = yy3d[k][j][i0];
            zz2d[k][j] = zz3d[k][j][i0];
        }
    }

    if(my_id == 0){
        printf("Write OCFDYZ-Mesh.dat\n");
        fp = fopen("OCFDYZ-Mesh.dat", "w");

        fprintf(fp, "variables=x,y,z\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, ny, nz);
    }

    for(int n = 0; n < n_processe; n++){
        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    fprintf(fp, "%15.6f%15.6f%15.6f\n", xx2d[k][j], yy2d[k][j], zz2d[k][j]);
                }
            }
        }

        if(my_id != 0){
            MPI_Send(x2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0) fclose(fp);
}

void Write_dataxz2d_cf_double_cone(int i0){//针对顿锥问题的后处理，写出表面摩阻
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk, us1, us2, h1, h2, uy, P, xp, yp, zp;
    double *cf0 = (double*)malloc(nx*sizeof(double));
    double *Tk0 = (double*)malloc(nx*sizeof(double));

    double *Ut = (double*)malloc(nx*sizeof(double));
    double *Ret = (double*)malloc(nx*sizeof(double));
    double *up = (double*)malloc(nz*sizeof(double));
    double *uvd = (double*)malloc(nz*sizeof(double));

    double (*zh2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//格点距离壁面距离
    double (*us2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//沿壁面流向的经过周向平均后的速度

    double (*d2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*cf2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*Tk2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*T2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    int m=391;     //周向j=m的截面

    average_data_xz(T2d, T);    //周向平均
    average_data_xz(d2d, d);

    comput_zh_us_compress_ramp(zh2d, us2d);

    Write_OCFDYZ_Mesh(i0);

    if(my_id == 0){

        for(int i = 0; i < nx; i++){

            us1 = us2d[1][i];
            us2 = us2d[2][i];

            h1 = zh2d[1][i];
            h2 = zh2d[2][i];

            uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);//展向速度梯度之和

            cf0[i] = 2*Amu*uy;

            Tk0[i] = 2*Amu*Cp/Pr*(T2d[1][i] - T2d[0][i])/h1;//计算展向热流和
        }

        printf("Write cf2d.dat\n");

        fp = fopen("cf2d.dat", "w");
        fprintf(fp, "variables=x,cf,Tk,pw\n");
        fprintf(fp, "zone i=%d\n", nx);
        for(int i = 0; i < nx; i++){
            P = d2d[1][i]*T2d[1][i];
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][504][i], cf0[i], Tk0[i], P);    //非均匀网格j的取值需要改  
        }
        fclose(fp);

//--------------------------------------------------------------------------------------
        for(int i = 0; i < nx; i++){
            Ut[i] = sqrt( fabs(cf0[i]) / (2*d2d[0][i]) );//摩擦速度
            Ret[i] = d2d[0][i]*Ut[i]/Amu;//粘性尺度倒数
        }

        fp = fopen("xyzp.dat", "w");
        fprintf(fp, "variables=x,xplus,yplus,zplus\n");
        fprintf(fp, "zone i=%d\n", nx-1);
        for(int i = 1; i < nx; i++){
            xp = (xx3d[0][m][i] - xx3d[0][m][i-1])*Ret[i];//Ut[i]/Amu;   //非均匀网格j的取值需要改
            yp = abs((yy3d[0][m][i] - yy3d[0][m][i]))*Ret[i];//Ut[i]/Amu;
            zp = 0.01*Ret[i];  // Ut[i]/Amu;

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][m][i],xp, yp, zp);
        }
        fclose(fp);


        printf("Write one-dimension profiles\n");

        double zp;
        printf("i0 is %d, Axx is %lf\n", i0, xx3d[0][0][i0]);    //i0需要在调用的时候赋值

        up[0] = 0; 
        uvd[0] = 0;

        fp = fopen("U1d.dat", "w");
        fprintf(fp, "variables=yp,up,uvd,u_log\n");
        fprintf(fp, "zone i=%d\n", nz-2);
        for(int k = 1; k < nz-1; k++){
            zp = zh2d[k][i0]*Ret[i0];
            up[k] = us2d[k][i0]/Ut[i0];
            uvd[k] = uvd[k-1] + sqrt(d2d[k][i0]/d2d[0][i0])*(up[k] - up[k-1]);

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", zp, up[k], uvd[k], 2.44*log(zp)+5.1);
        }
        fclose(fp);

//---------------------------------------------------------------------------------------

      /*  printf("Write delt\n");
        double delt0, delt1, delt2;
        int z0;

        fp = fopen("delta.dat", "w");

        for(int i = 0; i < nx; i++){
            delt1 = 0;
            delt2 = 0;
            for(int k = 0; k < nz; k++){
                if(us2d[k][i] > 0.99){
                    z0 = k-1;
                    goto end_comput_delt;
                }
            }

            end_comput_delt:;

            delt0 = zh2d[z0][i];//速度边界层厚度

            for(int k = 1; k <= z0; k++){
                delt1 += (1 - d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i]))*(zh2d[k][i] - zh2d[k-1][i]);
                delt2 += d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i])*(1 - us2d[k][i]/us2d[z0][i])*(zh2d[k][i] - zh2d[k-1][i]); 
            }
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], delt0, delt1, delt2);
        }
        fclose(fp);*/
    }
}


void Write_dataxy2d_cf_compress_ramp(int i0){//针对压缩折角问题的后处理，写出表面摩阻
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk, us1, us2, h1, h2, uy, P;
    double *cf0 = (double*)malloc(nx*sizeof(double));
    double *Tk0 = (double*)malloc(nx*sizeof(double));

    double *Ut = (double*)malloc(nx*sizeof(double));
    double *Ret = (double*)malloc(nx*sizeof(double));
    double *up = (double*)malloc(ny*sizeof(double));
    double *uvd = (double*)malloc(ny*sizeof(double));

    double (*yh2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));//格点距离壁面距离
    double (*us2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));//延壁面水平方向投影过的流向速度

    double (*d2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*u2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*v2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*T2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));

    average_data_xy(u2d, u);
    average_data_xy(v2d, v);
    average_data_xy(T2d, T);
    average_data_xy(d2d, d);


    if(my_id == 0){
        
        comput_yh_us_compress_ramp(yh2d, us2d, u2d, v2d);

        for(int i = 0; i < nx; i++){

            us1 = us2d[1][i];
            us2 = us2d[2][i];

            h1 = yh2d[1][i];
            h2 = yh2d[2][i];

            uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);//展向速度梯度之和

            cf0[i] = 2*Amu*uy;

            Tk0[i] = 2*Amu*Cp/Pr*(T2d[1][i] - T2d[0][i])/h1;//计算展向热流和
        }

        printf("Write cf2d.dat\n");

        fp = fopen("cf2d.dat", "w");
        fprintf(fp, "variables=x,cf,Tk\n");
        fprintf(fp, "zone i=%d\n", nx);
        for(int i = 0; i < nx; i++){
            P = d2d[1][i]*T2d[1][i];
            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], Tk0[i]);
        }
        fclose(fp);

//--------------------------------------------------------------------------------------
        for(int i = 0; i < nx; i++){
            Ut[i] = sqrt( fabs(cf0[i]) / (2*d2d[0][i]) );//摩擦速度
            Ret[i] = d2d[0][i]*Ut[i]/Amu;//粘性尺度倒数
        }

        printf("Write one-dimension profiles\n");

        double yp;
        printf("i0 is %d, Axx is %lf\n", i0, xx3d[0][0][i0]);

        up[0] = 0; 
        uvd[0] = 0;

        fp = fopen("U1d.dat", "w");
        fprintf(fp, "variables=yp,up,uvd,u_log\n");
        fprintf(fp, "zone i=%d\n", ny-2);
        for(int j = 1; j < ny-1; j++){
            yp = yh2d[j][i0]*Ret[i0];
            up[j] = us2d[j][i0]/Ut[i0];
            uvd[j] = uvd[j-1] + sqrt(d2d[j][i0]/d2d[0][i0])*(up[j] - up[j-1]);

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", yp, up[j], uvd[j], 2.44*log(yp)+5.1);
        }
        fclose(fp);

//---------------------------------------------------------------------------------------

        printf("Write delt\n");
        double delt0, delt1, delt2;
        int j0;

        fp = fopen("delta.dat", "w");

        for(int i = 0; i < nx; i++){
            delt1 = 0;
            delt2 = 0;
            for(int j = 0; j < ny; j++){
                if(us2d[j][i] > 0.99){
                    j0 = j-1;
                    goto end_comput_delt;
                }
            }

            end_comput_delt:;

            delt0 = yh2d[j0][i];//速度边界层厚度

            for(int j = 1; j <= j0; j++){
                delt1 += (1 - d2d[j][i]*us2d[j][i]/(d2d[j0][i]*us2d[j0][i]))*(yh2d[j][i] - yh2d[j-1][i]);
                delt2 += d2d[j][i]*us2d[j][i]/(d2d[j0][i]*us2d[j0][i])*(1 - us2d[j][i]/us2d[j0][i])*(yh2d[j][i] - yh2d[j-1][i]); 
            }
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], delt0, delt1, delt2);
        }
        fclose(fp);
    }
}

void Write_dataxyz3d_format(){
    int div = 10;
    int m = 7;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    if(my_id == 0) printf("Write dataxyz3d.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;

    x3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])x3d_buff;
    double (*yy3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])y3d_buff;
    double (*zz3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])z3d_buff;
    double (*ppd_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pd_buff;
    double (*ppu_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pu_buff;
    double (*ppv_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pv_buff;
    double (*ppw_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pw_buff;
    double (*ppT_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pT_buff;

    int tmp;

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < MP[m]; i++){
                tmp = MP_offset[m] + i;
                xx3d_buff[k][j][i] = xx3d[k][j][tmp];
                yy3d_buff[k][j][i] = yy3d[k][j][tmp];
                zz3d_buff[k][j][i] = zz3d[k][j][tmp];
                ppd_buff[k][j][i]  = d[k][j][tmp];
                ppu_buff[k][j][i]  = u[k][j][tmp];
                ppv_buff[k][j][i]  = v[k][j][tmp];
                ppw_buff[k][j][i]  = w[k][j][tmp];
                ppT_buff[k][j][i]  = T[k][j][tmp];
            }
        }
    }

    if(my_id == 0){
        sprintf(filename, "data07.dat");
        fp = fopen(filename, "w");
        fclose(fp);
    }

    for(int n = 0; n < n_processe; n++){

        sprintf(filename, "data07.dat");

        if(my_id == 0){
            fp = fopen(filename, "a");
            if(n == 0){
                fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
                fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], 2*ny-1, nz);
            }
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    for(int i = 0; i < MP[m]; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], 
                            ppw_buff[k][j][i], ppT_buff[k][j][i]);
                    }
                }
                for(int j = ny-2; j >= 0; j--){
                    for(int i = 0; i < MP[m]; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            -yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], 
                            ppw_buff[k][j][i], ppT_buff[k][j][i]);
                    }

                }
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}



//void Write_grid(){
//    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
//    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
//    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
//
//    printf("Write grid.dat\n");
//    if(my_id == 0){
//        fp = fopen("grid.dat", "w");
//        fprintf(fp, "variables=x,y,z\n");
//        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);
//    }
//
//    for(int n = 0; n < n_processe; n++){
//        
//        if(my_id == 0){
//            for(int k = 0; k < NPZ[n]; k++){
//                for(int j = 0; j < ny; j++){
//                    for(int i = 0; i < nx; i++){
//                        fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[k][j][i], yy3d[k][j][i], zz3d[k][j][i]);
//                    }
//                }
//            }
//        }
//
//        if(my_id != 0){
//            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//        }
//
//        if(my_id != n_processe-1){
//            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//        }
//    }
//
//    if(my_id == 0) fclose(fp);
//}

//void Write_data(){
//    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
//    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
//    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
//
//    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
//    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
//    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
//    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
//    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
//
//    fp = fopen("opencfd.format", "w");
//
//    fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
//    fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);
//    for(int k = 0; k < nz; k++){
//        for(int j = 0; j < ny; j++){
//            for(int i = 0; i < nx; i++){
//                fprintf(fp, "%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf\n", 
//                xx3d[k][j][i], yy3d[k][j][i], zz3d[k][j][i], 
//                d[k][j][i], u[k][j][i], v[k][j][i],
//                w[k][j][i], T[k][j][i]);
//            }
//        }
//    }
//
//    fclose(fp);
//}


void Finalize(){
    free(x3d);
    free(y3d);
    free(z3d);

    free(head);
    free(pd);
    free(pu);
    free(pv);
    free(pw);
    free(pT);
}
