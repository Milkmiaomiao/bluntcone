#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "mpi.h"
#include "pthread.h"

#define PI 3.141592653589793
//=========================需要修改的参数（mm）============================
#define xtransition 190.0
#define m1 16000    //6884(600mm for Rn=10mm)  //  256(70mm) 448(85mm for Rn=1mm)  640（100mm for Rn=1mm）
#define m2 21538   //13142(1100mm for Rn=10mm)   Rn=1mm{ 3702(200mm)   3200(300mm)   4480(400mm)}

// for output Q
#define i_step 4
#define j_step 4
#define k_step 1
#define i_begin 10013    //  6884(600mm for Rn=10mm)      640(100mm for Rn=1mm)
#define i_end   12516  //   13142(1100mm for Rn=10mm)    3702(340mm for Rn=1mm)
#define j_begin 0 
#define j_end   4500 //  750(30degree for Rn=10mm)     484(30degree for Rn=1mm)
#define k_begin 0
#define k_end   99

// for disturbance output
#define i_begin_disturbance 10013   
#define i_end_disturbance   12516


//for multi-streamwise station for Van Driest transformed velocity profiles
#define i1 12923  // 640(100mm for Rn=1mm)
#define i2 14154  // 896(120mm for Rn=1mm)
#define i3 14769  // 1152(140mm for Rn=1mm)
#define i4 15384  // 1408(160mm for Rn=1mm)
#define i5 16000  // 1664(180mm for Rn=1mm)
#define i6 16615  // 1920(200mm for Rn=1mm)
#define i7 17231  // 2176(220mm for Rn=1mm)
#define i8 17846  // 2132(240mm for Rn=1mm)
#define i9 18461  // 2688(260mm for Rn=1mm)
#define i10 19077  // 2944(280mm for Rn=1mm)
#define i11 19692  // 3200(300mm for Rn=1mm)
#define i12 20307  // 3456(320mm for Rn=1mm)
#define i13 21538  // 3712(340mm for Rn=1mm)
#define i14 22769  // 3968(360mm for Rn=1mm)


//for FIK
#define xe 9599   //800mm for Rn=1mm
#define nke 80
//==================================================================


FILE *fp;
FILE *fp1;
FILE *fp2;

MPI_Status status;

char  str[2000];
int init[3];
int my_id, n_processe;
int nx, ny, nz, NZ, *NPZ, *NP, *head;
int *NPZ_in, *NP_in;
double Re, Ama, Gamma, Pr, T_Ref, Tw, Amu, Cp, hh, tmp, p00;
double tmp1, tmp2,tmp3, tmp4, tmp5, tmp6, tmp7;
double *x3d, *y3d, *z3d;
double *pd, *pu, *pv, *pw, *pT, *pQ;
double *pd0, *pu0, *pv0, *pw0, *pT0;
double *ptmp, *pu_F, *pv_F, *pw_F;
double *pd_pur, *pu_pur, *pv_pur, *pw_pur, *pT_pur;
double *pu_Fpur, *pv_Fpur;
double *du, *dv;

double *pT02d, *pd02d, *pu02d, *pv02d, *pw02d, *pP02d;
double *pT_pur2d, *pd_pur2d, *pu_pur2d, *pv_pur2d, *pw_pur2d, *pP_pur2d;
double *pzh2d, *pxh2d, *pus2d, *puv2d, *puw2d;
double *pus_pur2d, *puv_pur2d;
double *pAmu_pur, *pAmu_ave;

double *CfB, *CfB_in, *CfV, *CfV_in, *CfT, *CfT_in, *CfM, *CfM_in, *CfD1,*CfD1_in,*CfD2,*CfD2_in,*CfD3,*CfD3_in,*CfD4,*CfD4_in;
double *CfV_new;

int *pboundary_k, *pentropy_k;





void mpi_init(int *Argc, char ***Argv);
void Data_malloc();
void Read_parameter();
void Read_mesh();
void Read_data();
void Read_data_Favreaverage();
void Read_data_average();
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
void comput_zh_us_bluntcone(double (*zh3d)[ny][nx], double (*us3d)[ny][nx]);
void comput_zh_xh_us_uv(double (*zh2d)[nx], double (*xh2d)[nx], double (*us2d)[nx], double (*uv2d)[nx], double (*uw2d)[nx]);
void compute_uspur_uvpur( double (*us2d)[nx], double (*uv2d)[nx]);
void average_data_xz_NZ(double (*data2d)[nx], double (*data3d)[ny][nx]);
//void Write_dataxy2d_yp();
void read_and_output_Q();
void find_boundary_and_entropy();
void compute_and_output_3d_disturbance();
void compute_for_CfV();
//void Favre_average_Favredata();
void Favre_average();
void twofold_FIK_decomposition();   //摩阻分解
void RD_decomposition();
void Finalize();

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();
    Data_malloc();

    Read_mesh();
    Read_data();
    Read_data_Favreaverage();
    Favre_average_Favredata();
    //Read_data_average();

    //Write_data_new();
    //Write_data_new_extension();

    //Write_cf2d_format();
    //Write_dataxy2d_first_format();
    //Write_grid_format();
    //Write_data_format();
    //Write_cf3d_format();
    // Write_datayz2d_format();
    // Write_dataxz2d_format();
    //Write_datayz2d1_format();
    //Write_dataxyz3d_format();

    //Write_flow1d_inlet();

    //Write_dataxy2d1_format();
    //Write_dataxy2d_cf_compress_ramp(800);
    //Write_dataxy2d_yp();


    
        //Write_cf2d_format();
        //Write_dataxy2d_first_format_total();
        //Write_cf3d_format_total();
        //Write_datayz2d_format();
        //Write_dataxz2d_format();
        //Write_dataxy2d_first_format_total();
        //Write_dataxyz3d_format();
        //read_and_output_Q();
        // compute_and_output_3d_disturbance();
        // find_boundary_and_entropy();
        // compute_for_CfV();
        // Favre_average();
        // twofold_FIK_decomposition();
        //Write_dataxz2d_cf_double_cone(6000);   // 6000(x=518mm,Rn=1mm)
    


    Finalize();

    return 0;
}

void mpi_init(int *Argc , char *** Argv){

	MPI_Init(Argc, Argv);

    MPI_Comm_rank(MPI_COMM_WORLD , &my_id);
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_processe);

    //printf("n_processe is %d\n",n_processe);

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
    }
//-----------------------------------------------------------------------------------------------------
    int tmp1[3];
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
    }

    MPI_Bcast(tmp1, 3, MPI_INT, 0, MPI_COMM_WORLD);
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
    }
//-------------------------------------------------------------------------------------------

    Amu = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(Tw*Tw*Tw)/(110.4/T_Ref + Tw);
    Cp = 1.0/((Gamma - 1)*Ama*Ama);   /*原来Cp = Gamma/((Gamma - 1)*Ama*Ama);*/
    p00 = 1.0/(Gamma*Ama*Ama);

//-------------------------------------------------------------------------------------------

    NZ = nz/n_processe;      //沿壁面法向划分多块

    if(my_id < nz%n_processe) NZ += 1;

    NPZ = (int*)malloc(n_processe * sizeof(int));
    NP = (int*)malloc(n_processe * sizeof(int));
    NPZ_in = (int*)malloc(n_processe * sizeof(int));
    NP_in = (int*)malloc(n_processe * sizeof(int));

    memset((void*)NPZ, 0, n_processe*sizeof(int));   /*为malloc新申请的连续内存进行初始化*/
    memset((void*)NP, 0, n_processe*sizeof(int));
    memset((void*)NPZ_in, 0, n_processe*sizeof(int));
    memset((void*)NP_in, 0, n_processe*sizeof(int));

    for(int i = 0; i < n_processe; i++){
        if(i < nz%n_processe){
            NPZ[i] = (int)nz/n_processe + 1;
        }else{
            NPZ[i] = (int)nz/n_processe;
        }
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];//偏移    
    }

    if(NP[n_processe-1] != nz-NPZ[n_processe-1]) printf("NP is wrong![debug]\n");

    for(int i = 0; i < n_processe; i++){
        NPZ_in[i] = NPZ[i] * nx;
        NP_in[i] = NP[i] *nx; 
    }
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


    head = (int*)malloc(5 * sizeof(int));

    pd = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pd);

    pu = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu);

    pv = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv);

    pw = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pw); 

    pT = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pT);

    // pQ = (double*)malloc(nx * ny * NZ * sizeof(double));
    // Malloc_Judge(pQ);

    pd0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pd0);

    pu0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu0);

    pv0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv0);

    pw0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pw0); 

    pT0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pT0);

    ptmp = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(ptmp);

    pu_F = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu_F);

    pv_F = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv_F);

    pw_F = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pw_F); 

    pd_pur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pd_pur);

    pu_pur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu_pur);

    pv_pur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv_pur);

    pw_pur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pw_pur); 

    pT_pur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pT_pur);

    pu_Fpur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu_Fpur);

    pv_Fpur = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv_Fpur);

    pboundary_k = (int*)malloc(nx * sizeof(int));

    pentropy_k = (int*)malloc(nx * sizeof(int));

    pd02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pd02d);

    pT02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pT02d);

    pu02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pu02d);

    pv02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pv02d);

    pw02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pw02d);

    pP02d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pP02d);

    pzh2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pzh2d);

    pxh2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pxh2d);

    pus2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pus2d);

    puv2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(puv2d);

    puw2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(puw2d);

    pus_pur2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(pus_pur2d);

    puv_pur2d = (double*)malloc(nx * nz * sizeof(double));
    Malloc_Judge(puv_pur2d);

    CfB = (double*)malloc(nx*sizeof(double));
    memset((void*)CfB, 0, nx*sizeof(double));

    CfB_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfB_in, 0, nx*sizeof(double));

    CfV = (double*)malloc(nx*nz*sizeof(double));
    memset((void*)CfV, 0, nx*nz*sizeof(double));

    CfV_in = (double*)malloc(nx*nz*sizeof(double));
    memset((void*)CfV_in, 0, nx*nz*sizeof(double));

    CfV_new = (double*)malloc(nx*sizeof(double));
    memset((void*)CfV_new, 0, nx*sizeof(double));

    CfT = (double*)malloc(nx*sizeof(double));
    memset((void*)CfT, 0, nx*sizeof(double));
    
    CfT_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfT_in, 0, nx*sizeof(double));
    
    CfM = (double*)malloc(nx*sizeof(double));
    memset((void*)CfM, 0, nx*sizeof(double));
    
    CfM_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfM_in, 0, nx*sizeof(double));

    CfD1 = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD1, 0, nx*sizeof(double));

    CfD1_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD1_in, 0, nx*sizeof(double));
    
    CfD2 = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD2, 0, nx*sizeof(double));
    
    CfD2_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD2_in, 0, nx*sizeof(double));
    
    CfD3 = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD3, 0, nx*sizeof(double));
    
    CfD3_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD3_in, 0, nx*sizeof(double));
    
    CfD4 = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD4, 0, nx*sizeof(double));
    
    CfD4_in = (double*)malloc(nx*sizeof(double));
    memset((void*)CfD4_in, 0, nx*sizeof(double));

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

    MPI_Barrier(MPI_COMM_WORLD);
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

    MPI_Barrier(MPI_COMM_WORLD);
}


void Read_data_Favreaverage(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "opencfd.Favreaverage", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file); //MPI_File_open(MPI_COMM_WORLD, "opencfd.ana", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_read_all(tmp_file, init, 1, MPI_INT, &status);             
    MPI_File_read_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, ptmp+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ u ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pu_F+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ v ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pv_F+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ w ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pw_F+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ T ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, ptmp+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
    printf("READ average data OK\n");

    MPI_Barrier(MPI_COMM_WORLD);
}



void Read_data_average(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "opencfd.average", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file); //MPI_File_open(MPI_COMM_WORLD, "opencfd.ana", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_read_all(tmp_file, init, 1, MPI_INT, &status);             
    MPI_File_read_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pd0+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ u ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pu0+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ v ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pv0+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ w ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pw0+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ T ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pT0+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
    printf("READ average data OK\n");

    MPI_Barrier(MPI_COMM_WORLD);
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
    int div = 10;
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
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

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
        /*for(int j = ny-2; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][ny-1][i] - T[0][ny-1][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                -yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }*/

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

    double (*zh3d)[ny][nx] = (double(*)[ny][nx])malloc(nx*ny*NZ*sizeof(double));
    double (*us3d)[ny][nx] = (double(*)[ny][nx])malloc(nx*ny*NZ*sizeof(double));

    double cf, Tk, hh, us1, us2, h1, h2, uy;

    int mx;

    mx = m2-m1+1;  //流向输出网格点数

    comput_zh_us_bluntcone(zh3d,us3d);

    if(my_id == 0){
        /*printf("Write datawall.dat\n");      //有需要时可以再开启，输出全锥的壁面摩阻热流

        fp = fopen("datawall.dat", "w");
        fprintf(fp, "variables=x,y,z,cf,Tk\n"); //d,u,v,w,T,
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                //hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                //          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                //          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));
                us1 = us3d[1][j][i];
                us2 = us3d[2][j][i];

                h1 = zh3d[1][j][i];
                h2 = zh3d[2][j][i];

                uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);

                cf = 2*Amu*uy;

                Tk = 2*Amu*Cp/Pr*(T[1][j][i] - T[0][j][i])/h1;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][j][i],  //%15.6f%15.6f%15.6f%15.6f%15.6f
                yy3d[0][j][i], zz3d[0][j][i], cf, Tk); //, d[0][j][i], u[0][j][i], v[0][j][i], w[0][j][i], T[0][j][i]
            }
        }

        fclose(fp);*/
    

        printf("Write datawall-seta-x.dat\n");

        fp = fopen("datawall-seta-x.dat", "w");
        fprintf(fp, "variables=x,seta,cf,Tk\n"); //d,u,v,w,T,
        fprintf(fp, "zone i=%d ,j=%d \n", mx, ny);
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < mx; i++){
               
                us1 = us3d[1][j][i+m1];
                us2 = us3d[2][j][i+m1];

                h1 = zh3d[1][j][i+m1];
                h2 = zh3d[2][j][i+m1];

                uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);

                cf = 2*Amu*uy;

                Tk = 2*Amu*Cp/Pr*(T[1][j][i+m1] - T[0][j][i+m1])/h1;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][j][i+m1],j * (360.0/ny), cf, Tk); 
            }
        }

        fclose(fp);
    }
}

void comput_zh_us_bluntcone(double (*zh3d)[ny][nx], double (*us3d)[ny][nx]){//计算点到壁面距离与平行壁面速度(不做周向平均)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);

    double length;
    double seta0 = 7.0*PI/180.0;
    
    double seta;

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);

    double *tmpx = (double*)malloc(nx*sizeof(double));
    double *tmpy = (double*)malloc(nx*sizeof(double));
    double *tmpz = (double*)malloc(nx*sizeof(double));

    if(my_id==0){
    for(int k=0; k<NZ; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){

            length = sqrt(pow(xx3d[k][0][i] - xx3d[0][0][i], 2) +                      //length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2));
                          pow(yy3d[k][0][i] - yy3d[0][0][i], 2) + pow(zz3d[k][0][i] - zz3d[0][0][i], 2) );                          

            zh3d[k][j][i] = length;
            }
        }
    }
       
    }
    /*MPI_Bcast(tmpx, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmpy, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmpz, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);*/

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

                us3d[k][j][i] = Vx[k][j][i];
            }
        }
    }

    /*if(my_id==0){
        
            for(int j=0; j<ny; j++){
                for(int i=0; i<nx; i++){
                    tmpx[i] = xx3d[0][0][i];
                    tmpy[i] = yy3d[0][0][i];
                    tmpz[i] = zz3d[0][0][i];    //没改完 
                }
            }
        
    }

    for(int k=0; k<NZ; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){

            length = sqrt(pow(xx3d[k][0][i] - tmpx[i], 2) +                      //length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2));
                          pow(yy3d[k][0][i] - tmpy[i], 2) + pow(zz3d[k][0][i] - tmpz[i], 2) );                          

            zh3d[k][j][i] = length;
            }
        }
    }*/
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
    int m=0;                  //选择j=m截面作为输出壁面摩阻的截面

    if(my_id == 0){
        printf("Write cf2d.dat\n");

        fp = fopen("cf2d_bottom.dat", "w");        //迎风面或背风面的一条摩阻曲线，非均匀网格下j要修改
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
    int div = 10;          //把流场沿流向均匀取10个站位
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

    int m=0;   //周向坐标j0=307

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


void Write_dataxy2d1_format(){                                //推测是写出壁面信息
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

void average_data_xz_NZ(double (*data2d)[nx], double (*data3d)[ny][nx]){
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

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            
            
            data2d[k][i] = data_buff[k][i];
           
            
        }
    }
}

void data_xz(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            
                data_buff[k][i] += data3d[k][0][i];
       
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

void get_data_xz(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            data_buff[k][i] = data3d[k][0][i];
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
    //double seta1 = 0.0;
    //int start_point = 8000;
    double seta;
    //double *seta = (double*)malloc(ny*sizeof(double));

    double *pxx_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*xx_buff)[nx] = (double (*)[nx])(pxx_buff);

    double *pyy_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*yy_buff)[nx] = (double (*)[nx])(pyy_buff);

    double *pzz_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*zz_buff)[nx] = (double (*)[nx])(pzz_buff);

    double *puu_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*uu_buff)[nx] = (double (*)[nx])(puu_buff);

    double (*xx2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*yy2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
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
            yy_buff[k][i] = yy3d[k][0][i];
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
                    yy2d[NP[n] + k][i] = yy_buff[k][i];
                    zz2d[NP[n] + k][i] = zz_buff[k][i];
                    us2d[NP[n] + k][i] = uu_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pxx_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pyy_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pzz_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(puu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pxx_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pyy_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pzz_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(puu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    for(int k=0; k<nz; k++){
        for(int i=0; i<nx; i++){

            length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) +                      //length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2));
                          pow(yy2d[k][i] - yy2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2) );                          

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

void output_Uvd(double (*zh2d)[nx], double (*us2d)[nx], double (*d2d)[nx], double (*Ret), double (*Ut), double (*up), double (*uvd), double (*z2d)[nx],int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);    
    double zp,zp2;
    char filename[100];
        up[0] = 0; 
        uvd[0] = 0;

        sprintf(filename, "U1d-%d.dat", (int)(xx3d[0][0][i0]));
        fp = fopen(filename, "w");

        fprintf(fp, "variables=yp,up,uvd,u_log,z,zp2\n");
        fprintf(fp, "zone i=%d\n", nz-2);
        for(int k = 1; k < nz-1; k++){
            zp = zh2d[k][i0]*Ret[i0];
            up[k] = us2d[k][i0]/Ut[i0];
            uvd[k] = uvd[k-1] + sqrt(d2d[k][i0]/d2d[0][i0])*(up[k] - up[k-1]);
            zp2 = (zh2d[k][i0] - zh2d[k-1][i0])*Ret[i0];       //沿壁面法向每一个网格的网格分辨率，横坐标应为z2d[k][i0]
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", zp, up[k], uvd[k], 2.44*log(zp)+5.1,z2d[k][i0],zp2 );
        }
        fclose(fp);
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

    double cf, Tk, us1, us2, h1, h2, uy, P, xp, yp, yp1, zp, zp1, ht1, ht2, Rex;
    double Cflaminar,Cflaminar1;

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

    //double (*Ma)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*p2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*u2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*v2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*w2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*h2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*ds)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*z2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*x2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    char filename[100];

    int m=0;     //周向j=m的截面

    average_data_xz(T2d, T);    //周向平均
    average_data_xz(d2d, d);

    data_xz(u2d, u);
    data_xz(v2d, v);
    data_xz(w2d, w);

    get_data_xz(z2d, zz3d);
    get_data_xz(x2d, xx3d);

    

    comput_zh_us_compress_ramp(zh2d, us2d);

    //Write_OCFDYZ_Mesh(i0);

    if(my_id == 0){

        printf("Write xx1d.dat\n");
        fp = fopen("xx1d.dat", "w");
        for(int i = 0; i < nx; i++){
            fprintf(fp, "%d%15.6f\n", i, xx3d[0][0][i]);
        }
        fclose(fp);

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
        fprintf(fp, "variables=x,cf,Tk,pw,Cflaminar,Cflaminar1\n");
        fprintf(fp, "zone i=%d\n", nx);
        for(int i = 0; i < nx; i++){
            P = d2d[1][i]*T2d[1][i];
            Rex = d2d[1][i] * us2d[1][i] * xx3d[0][0][i] / Amu;
            Cflaminar = 0.6641/sqrt(Rex);
            Cflaminar1 = 0.6641/sqrt(Re*xx3d[0][0][i]);
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], Tk0[i], P, Cflaminar, Cflaminar1);    //非均匀网格j的取值需要改  
        }
        fclose(fp);

//--------------------------------------------------------------------------------------
        for(int i = 0; i < nx; i++){
            Ut[i] = sqrt( fabs(cf0[i]) / (2*d2d[0][i]) );//摩擦速度
            Ret[i] = d2d[0][i]*Ut[i]/Amu;//粘性尺度倒数

        }

        fp = fopen("xyzp.dat", "w");
        fprintf(fp, "variables=x,xplus,yplus1,yplus,zplus,zplus1,Ret\n");
        fprintf(fp, "zone i=%d\n", nx-1);
        for(int i = 1; i < nx; i++){
            xp = (xx3d[0][m][i] - xx3d[0][m][i-1])*Ret[i];//Ret[i];   //非均匀网格j的取值需要改
            yp1 = abs((yy3d[0][m][i] - yy3d[0][m-1][i]))*Ret[i];//Ut[i]/Amu;
            yp = (2 * PI * sqrt( pow(yy3d[0][m][i] ,2) + pow(zz3d[0][m][i] ,2))/ny) * (Ret[i]);
            zp = 0.01*Ret[i];  // Ut[i]/Amu;
            zp1 = zh2d[1][i]*Ret[i];

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][m][i],xp, yp1, yp, zp, zp1, Ret[i]);
        }
        fclose(fp);

        /*fp = fopen("zp-%d.dat", i0,"w");         
        fprintf(fp, "variables=z,zplus\n");
        fprintf(fp, "zone i=%d\n", nz-1);
        double zp2;
        for(int i = 1; i < nz; i++){
            
            zp2 = (zh2d[i][i0] - zh2d[i-1][i0])*Ret[i0];

            fprintf(fp, "%15.6f%15.6f\n", z2d[i][i0],zp2);
        }
        fclose(fp);*/


        printf("Write one-dimension profiles\n");

        double zp2;
        //printf("i0 is %d, Axx is %lf\n", i0, xx3d[0][0][i0]);    //i0需要在调用的时候赋值
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d, i0);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i1);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i2);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i3);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i4);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i5);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i6);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i7);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i8);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i9);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i10);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i11);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i12);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i13);
        output_Uvd(zh2d, us2d, d2d, Ret, Ut, up, uvd,z2d,i14);
        //======================================== i0 ==========================================
        /*up[0] = 0; 
        uvd[0] = 0;

        sprintf(filename, "U1d-%d.dat", i0);
        fp = fopen(filename, "w");

        fprintf(fp, "variables=yp,up,uvd,u_log,z,zp2\n");
        fprintf(fp, "zone i=%d\n", nz-2);
        for(int k = 1; k < nz-1; k++){
            zp = zh2d[k][i0]*Ret[i0];
            up[k] = us2d[k][i0]/Ut[i0];
            uvd[k] = uvd[k-1] + sqrt(d2d[k][i0]/d2d[0][i0])*(up[k] - up[k-1]);
            zp2 = (zh2d[k][i0] - zh2d[k-1][i0])*Ret[i0];       //沿壁面法向每一个网格的网格分辨率，横坐标应为z2d[k][i0]
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", zp, up[k], uvd[k], 2.44*log(zp)+5.1,z2d[k][i0],zp2 );
        }
        fclose(fp);*/
    

//---------------------------------------------------------------------------------------

        printf("Write delt\n");
        double delt0, delt1, delt2;
        double KarmanSchoenherr,Blasius,Reseta,Resetainc,WhiteCf;
        double Amue,Amuw,Amuinf,Fseta,A,S;
        int z0,ze;

        fp = fopen("delta.dat", "w");
        fprintf(fp, "variables=x,delt0,delt1,delt2,KarmanSchoenherrCf,BlasiusCf,WhiteCf\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int i = 0; i < nx; i++){
            delt1 = 0;
            delt2 = 0;
            for(int k = 0; k < nz; k++){
                if(us2d[k][i] > 0.985 ){      //us2d[k][i] > 0.99 //*us2d[nz-1][0]
                    z0 = k-1;
                    goto end_comput_delt;
                }
            }

            end_comput_delt:;

            delt0 = zh2d[z0][i];//速度边界层厚度

            for(int k = 1; k <= z0; k++){
                delt1 += (1 - d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i]))*(zh2d[k][i] - zh2d[k-1][i]);
                delt2 += d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i])*(1 - us2d[k][i]/us2d[z0][i])*(zh2d[k][i] - zh2d[k-1][i]); //动量边界层厚度
            }

            for(int k = 0; k < nz; k++){                             //寻找动量厚度对应的z方向编号
                if(zh2d[k][i] > delt2 ){      //&& zh2d[k][i] >= delt2 
                    ze = k-1;
                    goto end_comput_ze;
                }
            }
            end_comput_ze:;

            A = sqrt((Gamma-1.0)/2.0 * pow(Ama* us2d[z0][i]/sqrt(T2d[z0][i]),2) * T2d[z0][i]/T2d[0][i]);

            S = (1.0/asin(A))*sqrt(T2d[0][i]/T2d[z0][i] - 1.0); 

            Amue = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(T2d[ze][i]*T2d[ze][i]*T2d[ze][i])/(110.4/T_Ref + T2d[ze][i]);
            
            Amuw = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(T2d[0][i]*T2d[0][i]*T2d[0][i])/(110.4/T_Ref + T2d[0][i]);

            WhiteCf = (0.455/(S*S)) * pow(log( (0.006/S) * (d2d[z0][i]*us2d[z0][i]*xx3d[0][0][i]/Amu - d2d[z0][i]*us2d[z0][i]*xtransition/Amu) * (Amue/Amuw) * sqrt(T2d[z0][i]/T2d[0][i]) ),-2);

            Reseta = d2d[ze][i]*us2d[ze][i]*delt2/Amue;

            Amuinf = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(T_Ref*T_Ref*T_Ref)/(110.4/T_Ref + T_Ref);

            Fseta = Amuinf/Amuw;

            Resetainc = Fseta * Reseta;

            KarmanSchoenherr = 1.0/(17.08 * pow(log10(Resetainc),2) + 25.11 * log10(Resetainc) + 6.012);

            Blasius = 0.026/pow(Resetainc,0.25);

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], delt0, delt1, delt2, KarmanSchoenherr, Blasius, WhiteCf);
        }
        fclose(fp);
    
    
        printf("Write Mach number\n");

        fp = fopen("Machnumber.dat", "w");
        fprintf(fp, "variables=x,z,Ma\n");
        fprintf(fp, "zone i=%d, j=%d\n", nx, nz);
        double Ma;

        for(int k = 0; k < nz; k++){
            for(int i = 0; i < nx; i++){
                Ma = Ama * us2d[k][i]/sqrt(T2d[k][i]);
                
                fprintf(fp, "%15.6f%15.6f%15.6f\n", x2d[k][i], z2d[k][i], Ma);
            }
        }

        fclose(fp);

        
        printf("Write boundary-layer thickness and entropy layer\n");
        
        fp = fopen("boundary-entropy.dat", "w");
        fprintf(fp, "variables=x,boundary,entropy\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int k = 0; k < nz; k++){
            for(int i = 0; i < nx; i++){
                
                p2d[k][i] = d2d[k][i] * T2d[k][i];
                ds[k][i] = (Gamma/(Gamma - 1.0)) * log(T2d[k][i]) - log(p2d[k][i]);
                
                ht1 = T2d[k][i]/(Gamma - 1.0) + 0.5 * (pow(u2d[k][i],2) + pow(v2d[k][i],2) + pow(w2d[k][i],2)) * Ama*Ama;
                ht2 = 1.0/(Gamma - 1.0) + 0.5 * Ama*Ama;
                h2d[k][i] = ht1/ht2;
            
            }
        }

        printf("finding boundary-layer...\n");
        double deltboundary[nx];
        int hz0;
        for(int i = 0; i < nx; i++){
            for(int k = 0; k < nz; k++){
                if( h2d[k][i] > 0.995 ){ 
                    hz0 = k-1;
                    goto end_comput_boundary;
                }
            }
            end_comput_boundary:;
            deltboundary[i] = zh2d[hz0][i];
            //fprintf(fp, "%15.6f%15.6f\n", xx3d[0][0][i], deltboundary);
        }
        printf("finding entropy-layer...\n");
        double deltentropy[nx];
        int hz1;
        for(int i = 0; i < nx; i++){
            for(int k = 0; k < nz; k++){
                if( ds[k][i] < 0.25*ds[0][i] ){ 
                    hz1 = k-1;
                    goto end_comput_entropy;
                }
            }
            end_comput_entropy:;
            deltentropy[i] = zh2d[hz1][i];
            //fprintf(fp, "%15.6f%15.6f\n", xx3d[0][0][i], deltboundary);
        }

        for(int i = 0; i < nx; i++){
            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[0][0][i], deltboundary[i],deltentropy[i]);
        }
        
        fclose(fp);


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
    //int div = 10;
    //int m = 7;
    //int m1=10,m2=20,m;
    
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
    /*int *MP = (int*)malloc(div * sizeof(int));
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

    MP_offset[div] = nx;*/

    /*x3d_buff = (double*)malloc(MP[m]* ny * NZ * sizeof(double));
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
    double (*ppT_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pT_buff;*/
    int mx ,my;
    mx= m2-m1+1;  //流向输出网格点数
    my = (j_end - j_begin)/j_step + 1;

    x3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    y3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    z3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    pd_buff  = (double*)malloc(mx * my * NZ * sizeof(double));
    pu_buff  = (double*)malloc(mx * my * NZ * sizeof(double));
    pv_buff  = (double*)malloc(mx * my * NZ * sizeof(double));
    pw_buff  = (double*)malloc(mx * my * NZ * sizeof(double));
    pT_buff  = (double*)malloc(mx * my * NZ * sizeof(double));

    double (*xx3d_buff)[my][mx] = (double(*)[my][mx])x3d_buff;
    double (*yy3d_buff)[my][mx] = (double(*)[my][mx])y3d_buff;
    double (*zz3d_buff)[my][mx] = (double(*)[my][mx])z3d_buff;
    double (*ppd_buff)[my][mx]  = (double(*)[my][mx])pd_buff;
    double (*ppu_buff)[my][mx]  = (double(*)[my][mx])pu_buff;
    double (*ppv_buff)[my][mx]  = (double(*)[my][mx])pv_buff;
    double (*ppw_buff)[my][mx]  = (double(*)[my][mx])pw_buff;
    double (*ppT_buff)[my][mx]  = (double(*)[my][mx])pT_buff;

    //int tmp;

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < my; j++){
            for(int i = 0; i < mx; i++){
                //tmp = MP_offset[m] + i;
                xx3d_buff[k][j][i] = xx3d[k][j+j_begin][i+m1];
                yy3d_buff[k][j][i] = yy3d[k][j+j_begin][i+m1];
                zz3d_buff[k][j][i] = zz3d[k][j+j_begin][i+m1];
                ppd_buff[k][j][i]  = d[k][j+j_begin][i+m1];
                ppu_buff[k][j][i]  = u[k][j+j_begin][i+m1];
                ppv_buff[k][j][i]  = v[k][j+j_begin][i+m1];
                ppw_buff[k][j][i]  = w[k][j+j_begin][i+m1];
                ppT_buff[k][j][i]  = T[k][j+j_begin][i+m1];
            }
        }
    }

    if(my_id == 0){
        sprintf(filename, "data3d-%d-%d.dat",(int)(xx3d[0][0][m1]),(int)(xx3d[0][0][m2]));
        fp = fopen(filename, "w");
        fclose(fp);
    }

    for(int n = 0; n < n_processe; n++){

        //sprintf(filename, "data3d-%d.dat",m);

        if(my_id == 0){
            fp = fopen(filename, "a");
            if(n == 0){
                fprintf(fp, "variables=x,y,z,d,u,v,w,T\n"); //,v,w
                fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", mx, my, nz);
            }
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < my; j++){
                    for(int i = 0; i < mx; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], ppw_buff[k][j][i], ppT_buff[k][j][i]); 
                    }
                }
                /*for(int j = ny-2; j >= 0; j--){
                    for(int i = 0; i < MP[m]; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            -yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], 
                            ppw_buff[k][j][i], ppT_buff[k][j][i]);
                    }

                }*/
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0) printf("Write dataxyz3d.dat OK \n");
}

void read_and_output_Q(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*Q)[ny][nx] = (double (*)[ny][nx])(pQ);

    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "Q.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file); 

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	/*MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_read_all(tmp_file, init, 1, MPI_INT, &status);             
    MPI_File_read_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);*/

    if(my_id == 0) printf("READ Q ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pQ+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    MPI_File_close(&tmp_file);
    printf("READ data Q OK\n");

    MPI_Barrier(MPI_COMM_WORLD);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pQ_buff;
    double *tmp_x, *tmp_y, *tmp_z;
    double hh;

    if(my_id == 0) printf("Write Q3d.dat\n");
    char filename[100];

    int mx,my,mz;
    mx = (i_end - i_begin)/i_step + 1;
    my = (j_end - j_begin)/j_step + 1;
    mz = (k_end - k_begin)/k_step + 1;

    x3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    y3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    z3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    pQ_buff  = (double*)malloc(mx * my * NZ * sizeof(double));
    tmp_x = (double*)malloc(mx * my * sizeof(double));
    tmp_y = (double*)malloc(mx * my * sizeof(double));
    tmp_z = (double*)malloc(mx * my * sizeof(double));
    
    double (*xx3d_buff)[my][mx] = (double(*)[my][mx])x3d_buff;
    double (*yy3d_buff)[my][mx] = (double(*)[my][mx])y3d_buff;
    double (*zz3d_buff)[my][mx] = (double(*)[my][mx])z3d_buff;
    double (*ppQ_buff)[my][mx]  = (double(*)[my][mx])pQ_buff;
    double (*ptmp_x)[mx] = (double(*)[mx])tmp_x;
    double (*ptmp_y)[mx] = (double(*)[mx])tmp_y;
    double (*ptmp_z)[mx] = (double(*)[mx])tmp_z;

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < my; j++){
            for(int i = 0; i < mx; i++){
                //tmp = MP_offset[m] + i;
                xx3d_buff[k][j][i] = xx3d[k][j+j_begin][i+i_begin];
                yy3d_buff[k][j][i] = yy3d[k][j+j_begin][i+i_begin];
                zz3d_buff[k][j][i] = zz3d[k][j+j_begin][i+i_begin];
                ppQ_buff[k][j][i]  = Q[k][j+j_begin][i+i_begin];
                
            }
        }
    }

    if(my_id == 0){
        
        for(int j = 0; j < my; j++){
            for(int i = 0; i < mx; i++){
                ptmp_x[j][i] = xx3d[0][j+j_begin][i+i_begin];
                ptmp_y[j][i] = yy3d[0][j+j_begin][i+i_begin];
                ptmp_z[j][i] = zz3d[0][j+j_begin][i+i_begin];
            }
        }    
        sprintf(filename, "data3d-Q-%d-%d-%d-%d-%d.dat",(int)(xx3d[0][0][i_begin]),(int)(xx3d[0][0][i_end]),i_step,j_step,k_step);
        fp = fopen(filename, "w");
        fclose(fp);
    }
    MPI_Bcast(ptmp_x, mx*my, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ptmp_y, mx*my, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ptmp_z, mx*my, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            fp = fopen(filename, "a");
            if(n == 0){
                fprintf(fp, "variables=x,y,z,Q,hh\n"); 
                fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", mx, my, nz);
            }
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < my; j++){
                    for(int i = 0; i < mx; i++){

                        hh = sqrt(pow(yy3d_buff[k][j][i]-ptmp_y[j][i],2)+pow(zz3d_buff[k][j][i]-ptmp_z[j][i],2));

                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppQ_buff[k][j][i],
                            hh); 
                    }
                }
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pQ_buff , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pQ_buff , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            
        }
    }

    if(my_id == 0) printf("Write data Q OK\n");
}


void compute_and_output_3d_disturbance(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double (*d0)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u0)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v0)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w0)[ny][nx] = (double (*)[ny][nx])(pw0);
    double (*T0)[ny][nx] = (double (*)[ny][nx])(pT0);

    double (*d_pur)[ny][nx] = (double (*)[ny][nx])(pd_pur);
    double (*u_pur)[ny][nx] = (double (*)[ny][nx])(pu_pur);
    double (*v_pur)[ny][nx] = (double (*)[ny][nx])(pv_pur);
    double (*w_pur)[ny][nx] = (double (*)[ny][nx])(pw_pur);
    double (*T_pur)[ny][nx] = (double (*)[ny][nx])(pT_pur);

    // int mx,my,mz;
    // mx = (i_end_disturbance - i_begin_disturbance)/i_step + 1;
    // my = (j_end - j_begin)/j_step + 1;
    // mz = (k_end - k_begin)/k_step + 1;
// 
    // if(my_id == 0) printf("mx = %d, my = %d, mz = %d",mx,my,mz);
// 
    // double *d_pur_tmp, *u_pur_tmp, *v_pur_tmp, *w_pur_tmp, *T_pur_tmp;
    // double *x3d_buff, *y3d_buff, *z3d_buff;
// 
    // double *d2d_pur_tmp, *u2d_pur_tmp, *T2d_pur_tmp;
    // double *x2d_buff, *y2d_buff, *z2d_buff;
// 
    // d_pur_tmp = (double*)malloc(mx * my * NZ * sizeof(double));
    // u_pur_tmp = (double*)malloc(mx * my * NZ * sizeof(double));
    // v_pur_tmp = (double*)malloc(mx * my * NZ * sizeof(double));
    // w_pur_tmp = (double*)malloc(mx * my * NZ * sizeof(double));
    // T_pur_tmp = (double*)malloc(mx * my * NZ * sizeof(double));
// 
    // x3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    // y3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
    // z3d_buff = (double*)malloc(mx * my * NZ * sizeof(double));
// 
    // d2d_pur_tmp = (double*)malloc(nx * NZ * sizeof(double));
    // u2d_pur_tmp = (double*)malloc(nx * NZ * sizeof(double));
    // T2d_pur_tmp = (double*)malloc(nx * NZ * sizeof(double));
// 
    // x2d_buff = (double*)malloc(nx * NZ * sizeof(double));
    // y2d_buff = (double*)malloc(nx * NZ * sizeof(double));
// 
    // double (*pd_pur_tmp)[my][mx] = (double (*)[my][mx])(d_pur_tmp);
    // double (*pu_pur_tmp)[my][mx] = (double (*)[my][mx])(u_pur_tmp);
    // double (*pv_pur_tmp)[my][mx] = (double (*)[my][mx])(v_pur_tmp);
    // double (*pw_pur_tmp)[my][mx] = (double (*)[my][mx])(w_pur_tmp);
    // double (*pT_pur_tmp)[my][mx] = (double (*)[my][mx])(T_pur_tmp);
// 
    // double (*xx3d_buff)[my][mx] = (double (*)[my][mx])(x3d_buff);
    // double (*yy3d_buff)[my][mx] = (double (*)[my][mx])(y3d_buff);
    // double (*zz3d_buff)[my][mx] = (double (*)[my][mx])(z3d_buff);
// 
    // double (*pd2d_pur_tmp)[nx] = (double (*)[nx])(d2d_pur_tmp);
    // double (*pu2d_pur_tmp)[nx] = (double (*)[nx])(u2d_pur_tmp);
    // double (*pT2d_pur_tmp)[nx] = (double (*)[nx])(T2d_pur_tmp);
// 
    // double (*xx2d_buff)[nx] = (double (*)[nx])(x2d_buff);
    // double (*yy2d_buff)[nx] = (double (*)[nx])(y2d_buff);
// 
    // double *z0;
// 
    // z0 = (double*)malloc(nx * sizeof(double));

    char filename[100];
    char filename1[100];

    if(my_id == 0) printf("start compute 3d disturbace");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                d_pur[k][j][i] = d[k][j][i] - d0[k][j][i];
                u_pur[k][j][i] = u[k][j][i] - u0[k][j][i];
                v_pur[k][j][i] = v[k][j][i] - v0[k][j][i];
                w_pur[k][j][i] = w[k][j][i] - w0[k][j][i];
                T_pur[k][j][i] = T[k][j][i] - T0[k][j][i];

            }

        }

    }



    // for(int k = 0; k < NZ; k++){
        // for(int j = 0; j < my; j++){
            // for(int i = 0; i < mx; i++){
// 
                // xx3d_buff[k][j][i] = xx3d[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // yy3d_buff[k][j][i] = yy3d[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // zz3d_buff[k][j][i] = zz3d[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
// 
                // pd_pur_tmp[k][j][i] = d_pur[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // pu_pur_tmp[k][j][i] = u_pur[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // pv_pur_tmp[k][j][i] = v_pur[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // pw_pur_tmp[k][j][i] = w_pur[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
                // pT_pur_tmp[k][j][i] = T_pur[k][j*j_step+j_begin][i*i_step+i_begin_disturbance];
// 
            // }
// 
        // }
// 
    // }
// 
    // for(int k = 0; k < NZ; k++){
        // 
        // for(int i = 0; i < nx; i++){
            // pd2d_pur_tmp[k][i] = d_pur[k][0][i];
            // pu2d_pur_tmp[k][i] = u_pur[k][0][i];
            // pT2d_pur_tmp[k][i] = T_pur[k][0][i];
            // 
            // xx2d_buff[k][i] = xx3d[k][0][i];
            // yy2d_buff[k][i] = sqrt(pow(yy3d[k][0][i],2) + pow(zz3d[k][0][i],2));
        // }
// 
    // }
// 
    // if(my_id == 0){
        // for(int i = 0; i < nx; i++){
            // z0[i] = yy2d_buff[0][i];
        // }
    // }
// 
    // MPI_Bcast(z0, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// 
    //  
    // sprintf(filename, "3d-disturbance-begain%d-end%d-istep%d-jstep%d-kstep%d.dat",(int)(xx3d[0][0][i_begin_disturbance]),(int)(xx3d[0][0][i_end_disturbance]),i_step,j_step,k_step);
    // sprintf(filename1, "2d-disturbance-j0.dat");
// 
    // for(int n = 0; n < n_processe; n++){
// 
        // if(my_id == 0){
            // fp = fopen(filename, "a");
            // fp1 = fopen(filename1, "a");
            // if(n == 0){
                // fprintf(fp, "variables=x,y,z,d,u,T\n"); 
                // fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", mx, my, nz);
                // 
                // fprintf(fp1, "variables=x,z,d,u,T\n"); 
                // fprintf(fp1, "zone i=%d ,j=%d \n", nx, nz);
            // }
    // 
            // for(int k = 0; k < NPZ[n]; k++){
                // for(int j = 0; j < my; j++){
                    // for(int i = 0; i < mx; i++){
// 
                        // fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            // yy3d_buff[k][j][i], zz3d_buff[k][j][i], pd_pur_tmp[k][j][i], pu_pur_tmp[k][j][i], pT_pur_tmp[k][j][i]); 
                    // }
                // }
            // }
// 
            // fclose(fp);
// 
            // for(int k = 0; k < NPZ[n]; k++){
                // 
                    // for(int i = 0; i < nx; i++){
// 
                        // fprintf(fp1, "%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx2d_buff[k][i],
                            // yy2d_buff[k][i]-z0[i], pd2d_pur_tmp[k][i], pu2d_pur_tmp[k][i], pT2d_pur_tmp[k][i]); 
                    // }
                // 
            // }
            // fclose(fp1);
// 
// 
        // }
// 
// 
        // if(my_id != 0){
            // MPI_Send(x3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(y3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(z3d_buff, mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(pd_pur_tmp , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(pu_pur_tmp , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(pT_pur_tmp , mx*my*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // 
            // MPI_Send(x2d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(y2d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(d2d_pur_tmp , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(u2d_pur_tmp , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(T2d_pur_tmp , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        // }
// 
        // if(my_id != n_processe-1){
            // MPI_Recv(x3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(y3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(z3d_buff, mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(pd_pur_tmp , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(pu_pur_tmp , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(pT_pur_tmp , mx*my*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // 
            // MPI_Recv(x2d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(y2d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // 
            // MPI_Recv(d2d_pur_tmp , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(u2d_pur_tmp , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(T2d_pur_tmp , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        // }
// 
        // 
    // }
// 
    // if(my_id == 0) printf("Write data disturbance OK\n");

}

void twofold_FIK_decomposition(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw0);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT0);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*T2d)[nx] = (double(*)[nx])(pT02d);
    double (*p2d)[nx] = (double(*)[nx])(pP02d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv02d);
    double (*w2d)[nx] = (double(*)[nx])(pw02d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv2d);
    double (*uw2d)[nx] = (double(*)[nx])(puw2d);

    // double (*uFavre_pur2d)[nx] = (double(*)[nx])(pu02d);
    // double (*vFavre_pur2d)[nx] = (double(*)[nx])(pv02d);

    double (*pCfB) = (double(*))(CfB);
    double (*pCfB_in) = (double(*))(CfB_in);
    double (*pCfV_new) = (double(*))(CfV_new);
    
    double (*pCfT) = (double(*))(CfT);
    double (*pCfT_in) = (double(*))(CfT_in);
    double (*pCfD4) = (double(*))(CfD4);
    double (*pCfD4_in) = (double(*))(CfD4_in);
    double (*pCfM) = (double(*))(CfM);
    double (*pCfM_in) = (double(*))(CfM_in);
    double (*pCfD2) = (double(*))(CfD2);
    double (*pCfD2_in) = (double(*))(CfD2_in);
    double (*pCfD1) = (double(*))(CfD1);
    double (*pCfD1_in) = (double(*))(CfD1_in);
    double (*pCfD3) = (double(*))(CfD3);
    double (*pCfD3_in) = (double(*))(CfD3_in);
    

    int (*boundary_k) = (int (*))(pboundary_k);

    double (*cf0) = (double(*))malloc(nx*sizeof(double)); 

    double cf, Tk, us1, us2, h1, h2, uuy,P, xp, yp, yp1, zp, zp1, ht1, ht2, Rex;
    //double uy, vx;

    // double *cf0 = (double*)malloc(nx*sizeof(double));
    // double *Tk0 = (double*)malloc(nx*sizeof(double));

    // double *Ut = (double*)malloc(nx*sizeof(double));
    // double *Ret = (double*)malloc(nx*sizeof(double));
    // double *up = (double*)malloc(nz*sizeof(double));
    // double *uvd = (double*)malloc(nz*sizeof(double));

    // double (*zh2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//格点距离壁面距离
    // double (*xh2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*us2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//沿壁面流向的经过周向平均后的速度
    // double (*uv2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    //double (*d2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*cf2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    // double (*Tk2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    //double (*T2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    // double (*p2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*u2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*v2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*w2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*h2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*z2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*x2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*uy)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*vx)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*wx)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*ux)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*px)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*sigmaxx)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*sigmaxx_dx)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    char filename[100];

    if(my_id == 0){
        printf("Compute CfB...\n");

        for(int j = 1; j < nz-1; j++){
            for(int i = 1; i < nx-1; i++){

                // uy[j][i] = (zh2d[j+1][i] * zh2d[j+1][i] * us2d[j][i] - zh2d[j][i] * zh2d[j][i] * us2d[j+1][i])/(
                //             zh2d[j+1][i] * zh2d[j+1][i] * zh2d[j][i] - zh2d[j][i] * zh2d[j][i] * zh2d[j+1][i]);
                
                // vx[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * uv2d[j][i] - xh2d[j][i] * xh2d[j][i] * uv2d[j][i+1])/(
                //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                // px[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * p2d[j][i] - xh2d[j][i] * xh2d[j][i] * p2d[j][i+1])/(
                            // xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                // px[j][i] = (xh2d[j][i] - xh2d[j][i-1])/((xh2d[j][i+1]-xh2d[j][i])*(xh2d[j][i+1]-xh2d[j][i-1]))*p2d[j][i+1] + 
                //            (xh2d[j][i+1] + xh2d[j][i-1])/((xh2d[j][i+1]-xh2d[j][i])*(xh2d[j][i] - xh2d[j][i-1]))*p2d[j][i] - 
                //            (xh2d[j][i+1]-xh2d[j][i])/((xh2d[j][i] - xh2d[j][i-1])*(xh2d[j][i+1]-xh2d[j][i-1]))*p2d[j][i-1];

                uy[j][i] = (us2d[j+1][i] - us2d[j][i])/(zh2d[j+1][i] - zh2d[j][i]);

                vx[j][i] = (uv2d[j][i+1] - uv2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                px[j][i] = (p2d[j][i+1] - p2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                ux[j][i] = (us2d[j][i+1] - us2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                wx[j][i] = (uw2d[j][i+1] - uw2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                // ux[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * us2d[j][i] - xh2d[j][i] * xh2d[j][i] * us2d[j][i+1])/(
                //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                // wx[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * uw2d[j][i] - xh2d[j][i] * xh2d[j][i] * uw2d[j][i+1])/(
                //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);
                
                sigmaxx[j][i] = 2.0 * Amu * (ux[j][i] - (1.0/3.0) * wx[j][i]);

                sigmaxx_dx[j][i] = (sigmaxx[j][i+1] - sigmaxx[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                // sigmaxx_dx[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * sigmaxx[j][i] - xh2d[j][i] * xh2d[j][i] * sigmaxx[j][i+1])/(
                //                     xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);
            
            }  
        }

        for(int i = 1; i < nx-1; i++){ 
            for(int j = 2; j < 1.3*boundary_k[i]; j++){
           
                // pCfB_in[i] = pCfB_in[i] + ((1.0/Re) * (Amu * uy[j][i] +  Amu * vx[j][i]) +
                //                            (1.0/Re) * (Amu * uy[j-1][i] +  Amu * vx[j-1][i])) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD2_in[i] = pCfD2_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) *px[j][i] +
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) *px[j-1][i]) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD3_in[i] = pCfD3_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) *(1.0/Re) * sigmaxx_dx[j][i] + 
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) *(1.0/Re) * sigmaxx_dx[j-1][i]  ) * (zh2d[j][i] - zh2d[j-1][i]/2.0) ;

                pCfB_in[i] = pCfB_in[i] + ((1.0/Re) * (Amu * uy[j][i] +  Amu * vx[j][i])) * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD2_in[i] = pCfD2_in[i] + ((zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) *px[j][i] ) * (zh2d[j][i] - zh2d[j-1][i]);

                //pCfD2_in[i] = pCfD2_in[i] + ((zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) *p2d[(int)(1.3*boundary_k[i])][i]*px[j][i] ) * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD3_in[i] = pCfD3_in[i] + ((zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) *(1.0/Re) * sigmaxx_dx[j][i]) * (zh2d[j][i] - zh2d[j-1][i]);
            }
        }

        for(int i = 1; i < nx-1; i++){

            pCfB[i] = (2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfB_in[i] ;

            pCfD2[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD2_in[i] ;

            pCfD3[i] = (2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD3_in[i] ;
        }

        for(int i = 0; i < nx; i++){

            us1 = us2d[1][i];
            us2 = us2d[2][i];

            h1 = zh2d[1][i];
            h2 = zh2d[2][i];

            //uuy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);
            uuy = (us1)/(h1);

            cf0[i] = 2*Amu*uuy;

        }


        printf("Write Cf.......dat\n");

        fp = fopen("cfFIK_kb=1.3boundary.dat","w");
        fprintf(fp, "variables=x,cf,cfB,cfV,cfT,cfD4,cfM,cfD1,cfD2,cfD3\n");
        fprintf(fp, "zone i=%d\n", nx-2);
        for(int i = 1; i < nx-1; i++){
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], pCfB[i], pCfV_new[i], pCfT[i], pCfD4[i], pCfM[i], pCfD1[i], pCfD2[i], pCfD3[i]);    //非均匀网格j的取值需要改  
        }
        fclose(fp);



    }

}

void comput_zh_xh_us_uv(double (*zh2d)[nx], double (*xh2d)[nx], double (*us2d)[nx], double (*uv2d)[nx], double (*uw2d)[nx]){//计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw0);

    double length, lengthx;
    double seta0 = 7.0*PI/180.0;
    //double seta1 = 0.0;
    //int start_point = 8000;
    double seta;
    //double *seta = (double*)malloc(ny*sizeof(double));

    double *pxx_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*xx_buff)[nx] = (double (*)[nx])(pxx_buff);

    double *pyy_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*yy_buff)[nx] = (double (*)[nx])(pyy_buff);

    double *pzz_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*zz_buff)[nx] = (double (*)[nx])(pzz_buff);

    double *puu_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*uu_buff)[nx] = (double (*)[nx])(puu_buff);

    double *pvv_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*vv_buff)[nx] = (double (*)[nx])(pvv_buff);

    double *pww_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*ww_buff)[nx] = (double (*)[nx])(pww_buff);

    

    double (*xx2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*yy2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
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
            yy_buff[k][i] = yy3d[k][0][i];
            zz_buff[k][i] = zz3d[k][0][i];

            uu_buff[k][i] = 0.0;
            vv_buff[k][i] = 0.0;
            ww_buff[k][i] = 0.0;

            for (int j=0; j<ny; j++)       //非均匀网格需要改
            {
                uu_buff[k][i] += Vx[k][j][i];
                vv_buff[k][i] += Vr[k][j][i];
                ww_buff[k][i] += Vseta[k][j][i];
            }

            uu_buff[k][i] /= ny;           //周向平均
            vv_buff[k][i] /= ny;
            ww_buff[k][i] /= ny;
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    xx2d[NP[n] + k][i] = xx_buff[k][i];
                    yy2d[NP[n] + k][i] = yy_buff[k][i];
                    zz2d[NP[n] + k][i] = zz_buff[k][i];
                    us2d[NP[n] + k][i] = uu_buff[k][i];
                    uv2d[NP[n] + k][i] = vv_buff[k][i];
                    uw2d[NP[n] + k][i] = ww_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pxx_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pyy_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pzz_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(puu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pvv_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pww_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pxx_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pyy_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pzz_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(puu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pvv_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pww_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0){
        printf("Compute length_h length_x");
        for(int k=0; k<nz; k++){
            for(int i=0; i<nx; i++){

                length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) +                      //length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2));
                              pow(yy2d[k][i] - yy2d[0][i], 2) + pow(zz2d[k][i] - zz2d[0][i], 2) );    

                lengthx = sqrt(pow(xx2d[k][i] - xx2d[k][0], 2) +                     
                              pow(yy2d[k][i] - yy2d[k][0], 2) + pow(zz2d[k][i] - zz2d[k][0], 2) );                      

                zh2d[k][i] = length;
                xh2d[k][i] = lengthx;
            }
        }
    }
}

void compute_uspur_uvpur( double (*us2d)[nx], double (*uv2d)[nx]){//计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu_pur);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv_pur);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw_pur);

    double length, lengthx;
    double seta0 = 7.0*PI/180.0;
    //double seta1 = 0.0;
    //int start_point = 8000;
    double seta;
    //double *seta = (double*)malloc(ny*sizeof(double));

    // double *pxx_buff = (double*)malloc(nx*NZ*sizeof(double));
    // double (*xx_buff)[nx] = (double (*)[nx])(pxx_buff);
// 
    // double *pyy_buff = (double*)malloc(nx*NZ*sizeof(double));
    // double (*yy_buff)[nx] = (double (*)[nx])(pyy_buff);
// 
    // double *pzz_buff = (double*)malloc(nx*NZ*sizeof(double));
    // double (*zz_buff)[nx] = (double (*)[nx])(pzz_buff);

    double *puu_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*uu_buff)[nx] = (double (*)[nx])(puu_buff);

    double *pvv_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*vv_buff)[nx] = (double (*)[nx])(pvv_buff);

    

    // double (*xx2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*yy2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    // double (*zz2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

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
            // xx_buff[k][i] = xx3d[k][0][i];
            // yy_buff[k][i] = yy3d[k][0][i];
            // zz_buff[k][i] = zz3d[k][0][i];

            uu_buff[k][i] = 0.0;
            vv_buff[k][i] = 0.0;

            for (int j=0; j<ny; j++)       //非均匀网格需要改
            {
                uu_buff[k][i] += Vx[k][j][i];
                vv_buff[k][i] += Vr[k][j][i];
            }

            uu_buff[k][i] /= ny;           //周向平均
            vv_buff[k][i] /= ny;
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    // xx2d[NP[n] + k][i] = xx_buff[k][i];
                    // yy2d[NP[n] + k][i] = yy_buff[k][i];
                    // zz2d[NP[n] + k][i] = zz_buff[k][i];
                    us2d[NP[n] + k][i] = uu_buff[k][i];
                    uv2d[NP[n] + k][i] = vv_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            // MPI_Send(pxx_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(pyy_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            // MPI_Send(pzz_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(puu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pvv_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            // MPI_Recv(pxx_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(pyy_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            // MPI_Recv(pzz_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(puu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pvv_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    
}

void find_boundary_and_entropy(){

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw0);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT0);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*T2d)[nx] = (double(*)[nx])(pT02d);
    double (*p2d)[nx] = (double(*)[nx])(pP02d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv02d);
    double (*w2d)[nx] = (double(*)[nx])(pw02d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv2d);
    double (*uw2d)[nx] = (double(*)[nx])(puw2d);

    double (*p0)[ny][nx] = (double(*)[ny][nx])malloc(nx*ny*NZ*sizeof(double));
    


    double ht1, ht2;
    //double *u2d_boundary, *v2d_boundary,*w2d_boundary;

    double (*u2d_boundary)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*v2d_boundary)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*w2d_boundary)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));


    double (*uvw2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*h2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*ds)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    int (*boundary_k) = (int (*))(pboundary_k);
    int (*entropy_k) = (int (*))(pentropy_k);

    double (*deltboundary) = (double(*))malloc(nx*sizeof(double));
    double (*deltentropy) = (double(*))malloc(nx*sizeof(double));
  

    char filename[100];

    int m=0;     //周向j=m的截面

    
    for(int k=0; k<NZ; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                p0[k][j][i] = (d[k][j][i] * T[k][j][i])*p00; //p0[k][j][i] = d[k][j][i] * T[k][j][i]
            }
        }
    }

    average_data_xz(T2d, T);    //周向平均
    average_data_xz(d2d, d);

    average_data_xz(p2d, p0);

    average_data_xz(u2d, u);
    average_data_xz(v2d, v);
    average_data_xz(w2d, w);

    get_data_xz(u2d_boundary, u);
    get_data_xz(v2d_boundary, v);
    get_data_xz(w2d_boundary, w);

    if(my_id == 0)printf("Compute zh2d xh2d us2d uv2d\n");
  

    comput_zh_xh_us_uv(zh2d, xh2d, us2d, uv2d, uw2d);
    if(my_id == 0)printf("Compute zh2d xh2d us2d uv2d OK\n");

    if(my_id == 0){

        // for(int j = 0; j < nz; j++){
        //     for(int i = 0; i < nx; i++){
        //         p2d[j][i] = d2d[j][i] * T2d[j][i];
        //     }
        // }

     printf("Write boundary-layer thickness and entropy layer\n");
        
        fp = fopen("boundary-entropy.dat", "w");
        fprintf(fp, "variables=x,boundary,entropy\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int k = 0; k < nz; k++){
            for(int i = 0; i < nx; i++){
                
                //p2d[k][i] = d2d[k][i] * T2d[k][i];
                ds[k][i] = (Gamma/(Gamma - 1.0)) * log(T2d[k][i]) - log(p2d[k][i]);
                //uvw2d[k][i] = sqrt(pow(u2d_boundary[k][i],2) + pow(v2d_boundary[k][i],2) + pow(w2d_boundary[k][i],2));
                
                ht1 = T2d[k][i]/(Gamma - 1.0) + 0.5 * (pow(u2d_boundary[k][i],2) + pow(v2d_boundary[k][i],2) + pow(w2d_boundary[k][i],2)) * Ama*Ama;
                ht2 = 1.0/(Gamma - 1.0) + 0.5 * Ama*Ama;
                h2d[k][i] = ht1/ht2;
            
            }
        }

        printf("finding boundary-layer...\n");
        
        
        for(int i = 0; i < nx; i++){
            for(int k = 0; k < nz; k++){
                if( h2d[k][i] > 0.995 ){ 
                    boundary_k[i] = k-1;
                    printf("boundary-layer_k = %d\n",boundary_k[i]);
                    goto end_comput_boundary;
                }
            }
            end_comput_boundary:;
            deltboundary[i] = zh2d[boundary_k[i]][i];
            //fprintf(fp, "%15.6f%15.6f\n", xx3d[0][0][i], deltboundary);
        }
        printf("finding entropy-layer...\n");
        
        
        for(int i = 0; i < nx; i++){
            for(int k = 0; k < nz; k++){
                if( ds[k][i] < 0.25*ds[0][i] ){ 
                    entropy_k[i] = k-1;
                    goto end_comput_entropy;
                }
            }
            end_comput_entropy:;
            deltentropy[i] = zh2d[entropy_k[i]][i];
            //fprintf(fp, "%15.6f%15.6f\n", xx3d[0][0][i], deltboundary);
        }

        for(int i = 0; i < nx; i++){
            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[0][0][i], deltboundary[i],deltentropy[i]);
        }
        
        fclose(fp);

        // fp1 = fopen("p2d.dat", "w");
        // fprintf(fp1, "variables=x,z,p\n");
        // fprintf(fp1, "zone i=%d, j=%d\n", nx,nz);

        // for(int j = 0; j < nz; j++){   
        //     for(int i = 0; i < nx; i++){
        //         fprintf(fp1, "%15.6f%15.6f%15.6f\n", xh2d[j][i], zh2d[j][i],p2d[j][i]);
        //     }
        // }     
        // fclose(fp1);
    }

    


}

void compute_for_CfV(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd_pur);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu_pur);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv_pur);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw_pur);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT_pur);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*T2d)[nx] = (double(*)[nx])(pT_pur2d);
    double (*p2d)[nx] = (double(*)[nx])(pP_pur2d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv_pur2d);
    double (*w2d)[nx] = (double(*)[nx])(pw_pur2d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus_pur2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv_pur2d);
    double (*us2d_0)[nx] = (double(*)[nx])(pus2d);

    double (*pCfV)[nx] = (double(*)[nx])(CfV);
    double (*pCfV_in)[nx] = (double(*)[nx])(CfV_in);
    double (*ppCfV) = (double(*))(CfV_new);

    double *pus2d_buff, *puv2d_buff;

    pus2d_buff = (double*)malloc(nx * NZ *sizeof(double));
    puv2d_buff = (double*)malloc(nx * NZ *sizeof(double));

    double (*us2d_buff)[nx] = (double (*)[nx])(pus2d_buff);
    double (*uv2d_buff)[nx] = (double (*)[nx])(puv2d_buff);

    double (*uy)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*vx)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    int (*boundary_k) = (int (*))(pboundary_k);

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);

    double seta;
    double seta0 = 7.0*PI/180.0;

    int jstep = 2500;
    
    
        if(my_id == 0)printf("Compute CfV...\n");

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

        for(int j = 0; j < ny/jstep - 1; j++){
            if(my_id == 0)printf("j = %d\n",j);

            for(int k = 0; k < NZ; k++ ){
                for(int i = 0; i < nx; i++){
                    us2d_buff[k][i] = Vx[k][j*jstep][i];
                    uv2d_buff[k][i] = Vr[k][j*jstep][i];
                }
            }


            
            if(my_id == 0)printf("MPI_Allgatherv ...\n");

            MPI_Allgatherv( pus2d_buff , nx*NZ , MPI_DOUBLE , pus_pur2d , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);
            MPI_Allgatherv( puv2d_buff , nx*NZ , MPI_DOUBLE , puv_pur2d , NPZ_in , NP_in, MPI_DOUBLE , MPI_COMM_WORLD);

            if(my_id == 0)printf("Compute ux  vy...\n");
        
            if(my_id == 0){

                for(int k = 1; k < nz-1; k++){
                    for(int i = 1; i < nx-1; i++){

                        // uy[k][i] = (zh2d[k+1][i] * zh2d[k+1][i] * us2d[k][i] - zh2d[k][i] * zh2d[k][i] * us2d[k+1][i])/(
                        //             zh2d[k+1][i] * zh2d[k+1][i] * zh2d[k][i] - zh2d[k][i] * zh2d[k][i] * zh2d[k+1][i]);

                        // vx[k][i] = (xh2d[k][i+1] * xh2d[k][i+1] * uv2d[k][i] - xh2d[k][i] * xh2d[k][i] * uv2d[k][i+1])/(
                        //             xh2d[k][i+1] * xh2d[k][i+1] * xh2d[k][i] - xh2d[k][i] * xh2d[k][i] * xh2d[k][i+1]);

                        uy[k][i] = (us2d[k+1][i] - us2d[k][i])/(zh2d[k+1][i] - zh2d[k][i]);

                        vx[k][i] = (uv2d[k][i+1] - uv2d[k][i])/(xh2d[k][i+1] - xh2d[k][i]);

                        pCfV_in[k][i] =  Amu * ( uy[k][i] +  vx[k][i]);
                    
                        pCfV[k][i] =  pCfV[k][i] + pCfV_in[k][i];
                    }  
                }

            }   


        }

        if(my_id == 0){
            
            for(int k = 1; k < nz-1; k++){
                for(int i = 1; i < nx-1; i++){
                
                    pCfV[k][i] = (pCfV[k][i] / (ny/jstep));   //周向平均  *(2.0/(d2d[boundary_k[i]][i] * u2d[boundary_k[i]][i]*u2d[boundary_k[i]][i] * zh2d[boundary_k[xe]][i])) 

                }
            }

            for(int i = 1; i < nx-1; i++){
                for(int k = 2; k < 1.3*boundary_k[i]; k++){
                
                
                    //ppCfV[i] = ppCfV[i] + (pCfV[k][i] + pCfV[k-1][i]) * (zh2d[k][i] - zh2d[k-1][i])/2.0  ;

                    ppCfV[i] = ppCfV[i] + pCfV[k][i] * (zh2d[k][i] - zh2d[k-1][i]);
                    
                }
            }

            for(int i = 1; i < nx-1; i++){
                ppCfV[i] = ppCfV[i] * (2.0/(d2d[boundary_k[i]][i] * us2d_0[boundary_k[i]][i]*us2d_0[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i]));
            }

            printf("Compute CfV OK\n");
        }

}


void Favre_average(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);

    //============================for Favre average==============================
    // double (*pdu)[ny][nx] = (double (*)[ny][nx])(du);
    // double (*pdv)[ny][nx] = (double (*)[ny][nx])(dv);

    // double *pdu_F, *pdv_F, *pd0_F;

    // pdu_F = (double*)malloc(nx * NZ *sizeof(double));
    // pdv_F = (double*)malloc(nx * NZ *sizeof(double));
    // pd0_F = (double*)malloc(nx * NZ *sizeof(double));

    // double (*du_F)[nx] = (double (*)[nx])(pdu_F);
    // double (*dv_F)[nx] = (double (*)[nx])(pdv_F);
    // double (*d0_F)[nx] = (double (*)[nx])(pd0_F);

    // double (*u_pur)[ny][nx] = (double (*)[ny][nx])(pu_Fpur);
    // double (*v_pur)[ny][nx] = (double (*)[ny][nx])(pv_Fpur);

    //===========================================================================
    
    double (*d_pur)[ny][nx] = (double (*)[ny][nx])(pd_pur);
    double (*u_pur)[ny][nx] = (double (*)[ny][nx])(pu_pur);
    double (*v_pur)[ny][nx] = (double (*)[ny][nx])(pv_pur);
    double (*w_pur)[ny][nx] = (double (*)[ny][nx])(pw_pur);
    double (*T_pur)[ny][nx] = (double (*)[ny][nx])(pT_pur);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*d0)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u0)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v0)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w0)[ny][nx] = (double (*)[ny][nx])(pw0);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv02d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv2d);

    double (*pCfT) = (double(*))(CfT);
    double (*pCfT_in) = (double(*))(CfT_in);
    double (*pCfD4) = (double(*))(CfD4);
    double (*pCfD4_in) = (double(*))(CfD4_in);
    double (*pCfM) = (double(*))(CfM);
    double (*pCfM_in) = (double(*))(CfM_in);
    double (*pCfD1) = (double(*))(CfD1);
    double (*pCfD1_in) = (double(*))(CfD1_in);

    int (*boundary_k) = (int (*))(pboundary_k);

    double *pv_Favre0, *pu_Favre0;
    double *puv_Favre, *puu_Favre;
    double *pu_Favre, *pv_Favre;
    double *puv_2d,  *puu_2d;
    double *pv_2d0,  *pu_2d0 ,*pd00;
    double *pu_2d,  *pv_2d;

    pv_Favre0 = (double*)malloc(nx * ny * NZ *sizeof(double));
    pu_Favre0 = (double*)malloc(nx * ny * NZ *sizeof(double));
    puv_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    puu_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    pu_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    pv_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));

    puv_2d = (double*)malloc(nx * nz *sizeof(double));
    
    puu_2d = (double*)malloc(nx * nz *sizeof(double));

    pv_2d0 = (double*)malloc(nx * NZ *sizeof(double));
    
    pu_2d0 = (double*)malloc(nx * NZ *sizeof(double));
    pd00 = (double*)malloc(nx * NZ *sizeof(double));

    pu_2d = (double*)malloc(nx * nz *sizeof(double));
    
    pv_2d = (double*)malloc(nx * nz *sizeof(double));

    double (*v_Favre0)[ny][nx] = (double (*)[ny][nx])(pv_Favre0);
    double (*u_Favre0)[ny][nx] = (double (*)[ny][nx])(pu_Favre0);
    double (*uv_Favre)[ny][nx] = (double (*)[ny][nx])(puv_Favre);
    double (*uu_Favre)[ny][nx] = (double (*)[ny][nx])(puu_Favre);
    double (*u_Favre)[ny][nx] = (double (*)[ny][nx])(pu_Favre);
    double (*v_Favre)[ny][nx] = (double (*)[ny][nx])(pv_Favre);

    double (*uv_2d)[nx] = (double (*)[nx])(puv_2d);
    
    double (*uu_2d)[nx] = (double (*)[nx])(puu_2d);

    double (*v_2d0)[nx] = (double (*)[nx])(pv_2d0);
    
    double (*u_2d0)[nx] = (double (*)[nx])(pu_2d0);

    double (*d00)[nx] = (double (*)[nx])(pd00);

    double (*u_2d)[nx] = (double (*)[nx])(pu_2d);
    
    double (*v_2d)[nx] = (double (*)[nx])(pv_2d);

    double (*uux)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*uy)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*ux)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);

    double *pVx_0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVseta_0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVr_0 = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx_0)[ny][nx] = (double (*)[ny][nx])(pVx_0);
    double (*Vseta_0)[ny][nx] = (double (*)[ny][nx])(pVseta_0);
    double (*Vr_0)[ny][nx] = (double (*)[ny][nx])(pVr_0);

    double seta;
    double seta0 = 7.0*PI/180.0;

    if(my_id == 0) printf("start compute 3d Favre average");

    //============================for Favre average=======================================
    // average_data_xz(uv_2d, pdu);   //周向平均
    // average_data_xz(uv_2d, pdv);

    // average_data_xz_NZ(du_F,pdu);
    // average_data_xz_NZ(dv_F,pdv);

    // average_data_xz_NZ(d0_F,d0);

    // for(int k = 0; k < NZ; k++){        
    //     for(int i = 0; i < nx; i++){
    //         du_F[k][i] = du_F[k][i] / d0_F[k][i];
    //         dv_F[k][i] = dv_F[k][i] / d0_F[k][i];

    //     }
    // }

    // for(int k = 0; k < NZ; k++){
    //     for(int j = 0; j < ny; j++){
    //         for(int i = 0; i < nx; i++){
    //             u_pur[k][j][i] = u[k][j][i] - du_F[k][i];
    //             v_pur[k][j][i] = v[k][j][i] - dv_F[k][i];

    //         }

    //     }

    // }
    //====================================================================================
    

    for(int i=0; i<nx; i++){
        for(int k=0; k<NZ; k++){
            for(int j=0; j<ny; j++){
                seta=acos(zz3d[0][j][0]/sqrt(zz3d[0][j][0]*zz3d[0][j][0]+yy3d[0][j][0]*yy3d[0][j][0]));
    
                if(yy3d[0][j][0] < 0){
                    seta=2*PI-seta;
                }
                // Vr[k][j][i] = v_pur[k][j][i]*sin(seta) + w_pur[k][j][i]*cos(seta);     //径向速度，对应周向均匀网格，非均匀网格需要改，读seta文件？
                // Vseta[k][j][i] = v_pur[k][j][i]*cos(seta) - w_pur[k][j][i]*sin(seta);  //周向速度
                // Vx[k][j][i] = u_pur[k][j][i]*cos(seta0) + Vr[k][j][i]*sin(seta0);       //平行壁面速度
                // Vr[k][j][i] = -u_pur[k][j][i]*sin(seta0) + Vr[k][j][i]*cos(seta0);      //垂直于壁面的速度

                Vr_0[k][j][i] = v0[k][j][i]*sin(seta) + w0[k][j][i]*cos(seta);     //径向速度，对应周向均匀网格，非均匀网格需要改，读seta文件？
                Vseta_0[k][j][i] = v0[k][j][i]*cos(seta) - w0[k][j][i]*sin(seta);  //周向速度
                Vx_0[k][j][i] = u0[k][j][i]*cos(seta0) + Vr_0[k][j][i]*sin(seta0);       //平行壁面速度
                Vr_0[k][j][i] = -u0[k][j][i]*sin(seta0) + Vr_0[k][j][i]*cos(seta0);      //垂直于壁面的速度
            }
        }
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                v_Favre0[k][j][i] =  Vr_0[k][j][i] * d0[k][j][i];
                u_Favre0[k][j][i] =  Vx_0[k][j][i] * d0[k][j][i];

            }

        }

    }

    average_data_xz_NZ(v_2d0,v_Favre0);
    average_data_xz_NZ(u_2d0,u_Favre0);
    average_data_xz_NZ(d00,d0);

    for(int k = 0; k < NZ; k++){        
        for(int i = 0; i < nx; i++){
                v_2d0[k][i] = v_2d0[k][i] / d00[k][i];
                u_2d0[k][i] = u_2d0[k][i] / d00[k][i];

        }
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                v_Favre0[k][j][i] = Vr_0[k][j][i] - v_2d0[k][i];  //v_Favre0[k][j][i] = v_Favre0[k][j][i] - v_2d0[k][i];
                                                                      //u_Favre0[k][j][i] = u_Favre0[k][j][i] - u_2d0[k][i];
                u_Favre0[k][j][i] = Vx_0[k][j][i] - u_2d0[k][i];

            }

        }

    }



    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                uv_Favre[k][j][i] = v_Favre0[k][j][i] * u_Favre0[k][j][i] * d0[k][j][i];
                uu_Favre[k][j][i] = u_Favre0[k][j][i] * u_Favre0[k][j][i] * d0[k][j][i];

                u_Favre[k][j][i] = Vx_0[k][j][i] * d0[k][j][i];
                v_Favre[k][j][i] = Vr_0[k][j][i] * d0[k][j][i];

            }

        }

    }

    average_data_xz(uv_2d, uv_Favre);
    average_data_xz(uu_2d, uu_Favre);

    average_data_xz(u_2d, u_Favre);
    average_data_xz(v_2d, v_Favre);
    
    

    if(my_id == 0){
        for(int j = 0; j < nz; j++){
          for(int i = 0; i < nx; i++){
              u_2d[j][i] = u_2d[j][i] / d2d[j][i]; 
              v_2d[j][i] = v_2d[j][i] / d2d[j][i];    

            }

        }
        for(int j = 1; j < nz-1; j++){
                for(int i = 1; i < nx-1; i++){

                    uux[j][i] = (uu_2d[j][i+1] - uu_2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                    uy[j][i] = (u_2d[j+1][i] - u_2d[j][i])/(zh2d[j+1][i] - zh2d[j][i]);

                    ux[j][i] = (u_2d[j][i+1] - u_2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                    // uux[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * uu_2d[j][i] - xh2d[j][i] * xh2d[j][i] * uu_2d[j][i+1])/(
                    //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                    // uy[j][i] = (zh2d[j+1][i] * zh2d[j+1][i] * u_2d[j][i] - zh2d[j][i] * zh2d[j][i] * u_2d[j+1][i])/(
                    //             zh2d[j+1][i] * zh2d[j+1][i] * zh2d[j][i] - zh2d[j][i] * zh2d[j][i] * zh2d[j+1][i]);

                    // uy[j][i] = (zh2d[j][i]-zh2d[j-1][i])/((zh2d[j+1][i]-zh2d[j][i])*(zh2d[j+1][i]-zh2d[j-1][i]))*u_2d[j+1][i] + 
                    //            (zh2d[j+1][i]-zh2d[j-1][i])/((zh2d[j+1][i]-zh2d[j][i])*(zh2d[j][i]-zh2d[j-1][i]))*u_2d[j][i] - 
                    //            (zh2d[j+1][i]-zh2d[j][i])/((zh2d[j][i]-zh2d[j-1][i])*(zh2d[j+1][i]-zh2d[j-1][i]))*u_2d[j-1][i];

                    // ux[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * u_2d[j][i] - xh2d[j][i] * xh2d[j][i] * u_2d[j][i+1])/(
                    //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                    
                    
                }  
            }
        
        for(int i = 1; i < nx-1; i++){
            for(int j = 2; j < 1.3*boundary_k[i]; j++){
            
                
                // pCfT_in[i] = pCfT_in[i] + (uv_2d[j][i] + uv_2d[j-1][i])* (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD4_in[i] = pCfD4_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * uux[j][i] +
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) * uux[j-1][i])* (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfM_in[i] = pCfM_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * d2d[j][i] * v_2d[j][i] * uy[j][i] +
                //                            (zh2d[nz-1][i] - zh2d[j-1][i]) * d2d[j-1][i] * v_2d[j-1][i] * uy[j-1][i]) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD1_in[i] = pCfD1_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * d2d[j][i] * u_2d[j][i] * ux[j][i] +
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) * d2d[j-1][i] * u_2d[j-1][i] * ux[j-1][i]) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                pCfT_in[i] = pCfT_in[i] + uv_2d[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD4_in[i] = pCfD4_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * uux[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfM_in[i] = pCfM_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * d2d[j][i] * v_2d[j][i] * uy[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD1_in[i] = pCfD1_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * d2d[j][i] * u_2d[j][i] * ux[j][i] * (zh2d[j][i] - zh2d[j-1][i]);
            }
        }
        

            for(int i = 1; i < nx-1; i++){

                pCfT[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfT_in[i];

                pCfD4[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD4_in[i]; 

                pCfM[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfM_in[i];  

                pCfD1[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD1_in[i];
            }

        printf("Compute CfT CfD4 CfM OK\n");

    }


}



void Favre_average_Favredata(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    // double (*u)[ny][nx] = (double (*)[ny][nx])(pu_F);
    // double (*v)[ny][nx] = (double (*)[ny][nx])(pv_F);
    // double (*w)[ny][nx] = (double (*)[ny][nx])(pw_F);

    double (*d0)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u0)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v0)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w0)[ny][nx] = (double (*)[ny][nx])(pw0);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv02d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv2d);

    double (*pCfT) = (double(*))(CfT);
    double (*pCfT_in) = (double(*))(CfT_in);
    double (*pCfD4) = (double(*))(CfD4);
    double (*pCfD4_in) = (double(*))(CfD4_in);
    double (*pCfM) = (double(*))(CfM);
    double (*pCfM_in) = (double(*))(CfM_in);
    double (*pCfD1) = (double(*))(CfD1);
    double (*pCfD1_in) = (double(*))(CfD1_in);

    int (*boundary_k) = (int (*))(pboundary_k);

    double *pv_Favre0, *pu_Favre0;
    double *puv_Favre, *puu_Favre;
    double *pu_Favre, *pv_Favre;
    double *puv_2d,  *puu_2d;
    double *pv_2d0,  *pu_2d0 ,*pd00;
    double *pu_2d,  *pv_2d;

    pv_Favre0 = (double*)malloc(nx * ny * NZ *sizeof(double));
    pu_Favre0 = (double*)malloc(nx * ny * NZ *sizeof(double));
    puv_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    puu_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    pu_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));
    pv_Favre = (double*)malloc(nx * ny * NZ *sizeof(double));

    puv_2d = (double*)malloc(nx * nz *sizeof(double));
    
    puu_2d = (double*)malloc(nx * nz *sizeof(double));

    pv_2d0 = (double*)malloc(nx * NZ *sizeof(double));
    
    pu_2d0 = (double*)malloc(nx * NZ *sizeof(double));
    pd00 = (double*)malloc(nx * NZ *sizeof(double));

    pu_2d = (double*)malloc(nx * nz *sizeof(double));
    
    pv_2d = (double*)malloc(nx * nz *sizeof(double));

    double (*v_Favre0)[ny][nx] = (double (*)[ny][nx])(pv_Favre0);
    double (*u_Favre0)[ny][nx] = (double (*)[ny][nx])(pu_Favre0);
    double (*uv_Favre)[ny][nx] = (double (*)[ny][nx])(puv_Favre);
    double (*uu_Favre)[ny][nx] = (double (*)[ny][nx])(puu_Favre);
    double (*u_Favre)[ny][nx] = (double (*)[ny][nx])(pu_Favre);
    double (*v_Favre)[ny][nx] = (double (*)[ny][nx])(pv_Favre);

    double (*uv_2d)[nx] = (double (*)[nx])(puv_2d);
    
    double (*uu_2d)[nx] = (double (*)[nx])(puu_2d);

    double (*v_2d0)[nx] = (double (*)[nx])(pv_2d0);
    
    double (*u_2d0)[nx] = (double (*)[nx])(pu_2d0);

    double (*d00)[nx] = (double (*)[nx])(pd00);

    double (*u_2d)[nx] = (double (*)[nx])(pu_2d);
    
    double (*v_2d)[nx] = (double (*)[nx])(pv_2d);

    double (*uux)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*uy)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double (*ux)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);

    double *pVx_0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVseta_0 = (double*)malloc(nx * ny * NZ * sizeof(double));
    double *pVr_0 = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*Vx_0)[ny][nx] = (double (*)[ny][nx])(pVx_0);
    double (*Vseta_0)[ny][nx] = (double (*)[ny][nx])(pVseta_0);
    double (*Vr_0)[ny][nx] = (double (*)[ny][nx])(pVr_0);

    double seta;
    double seta0 = 7.0*PI/180.0;

    if(my_id == 0) printf("start compute 3d Favre average");

    //============================for Favre average=======================================
    // average_data_xz(uv_2d, pdu);   //周向平均
    // average_data_xz(uv_2d, pdv);

    // average_data_xz_NZ(du_F,pdu);
    // average_data_xz_NZ(dv_F,pdv);

    // average_data_xz_NZ(d0_F,d0);

    // for(int k = 0; k < NZ; k++){        
    //     for(int i = 0; i < nx; i++){
    //         du_F[k][i] = du_F[k][i] / d0_F[k][i];
    //         dv_F[k][i] = dv_F[k][i] / d0_F[k][i];

    //     }
    // }

    // for(int k = 0; k < NZ; k++){
    //     for(int j = 0; j < ny; j++){
    //         for(int i = 0; i < nx; i++){
    //             u_pur[k][j][i] = u[k][j][i] - du_F[k][i];
    //             v_pur[k][j][i] = v[k][j][i] - dv_F[k][i];

    //         }

    //     }

    // }
    //====================================================================================
    

    for(int i=0; i<nx; i++){
        for(int k=0; k<NZ; k++){
            for(int j=0; j<ny; j++){
                seta=acos(zz3d[0][j][0]/sqrt(zz3d[0][j][0]*zz3d[0][j][0]+yy3d[0][j][0]*yy3d[0][j][0]));
    
                if(yy3d[0][j][0] < 0){
                    seta=2*PI-seta;
                }
                // Vr[k][j][i] = v_pur[k][j][i]*sin(seta) + w_pur[k][j][i]*cos(seta);     //径向速度，对应周向均匀网格，非均匀网格需要改，读seta文件？
                // Vseta[k][j][i] = v_pur[k][j][i]*cos(seta) - w_pur[k][j][i]*sin(seta);  //周向速度
                // Vx[k][j][i] = u_pur[k][j][i]*cos(seta0) + Vr[k][j][i]*sin(seta0);       //平行壁面速度
                // Vr[k][j][i] = -u_pur[k][j][i]*sin(seta0) + Vr[k][j][i]*cos(seta0);      //垂直于壁面的速度

                Vr_0[k][j][i] = v0[k][j][i]*sin(seta) + w0[k][j][i]*cos(seta);     //径向速度，对应周向均匀网格，非均匀网格需要改，读seta文件？
                Vseta_0[k][j][i] = v0[k][j][i]*cos(seta) - w0[k][j][i]*sin(seta);  //周向速度
                Vx_0[k][j][i] = u0[k][j][i]*cos(seta0) + Vr_0[k][j][i]*sin(seta0);       //平行壁面速度
                Vr_0[k][j][i] = -u0[k][j][i]*sin(seta0) + Vr_0[k][j][i]*cos(seta0);      //垂直于壁面的速度
            }
        }
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                v_Favre0[k][j][i] =  Vr_0[k][j][i] * d0[k][j][i];
                u_Favre0[k][j][i] =  Vx_0[k][j][i] * d0[k][j][i];

            }

        }

    }

    average_data_xz_NZ(v_2d0,v_Favre0);
    average_data_xz_NZ(u_2d0,u_Favre0);
    average_data_xz_NZ(d00,d0);

    for(int k = 0; k < NZ; k++){        
        for(int i = 0; i < nx; i++){
                v_2d0[k][i] = v_2d0[k][i] / d00[k][i];
                u_2d0[k][i] = u_2d0[k][i] / d00[k][i];

        }
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                v_Favre0[k][j][i] = Vr_0[k][j][i] - v_2d0[k][i];  //v_Favre0[k][j][i] = v_Favre0[k][j][i] - v_2d0[k][i];
                                                                      //u_Favre0[k][j][i] = u_Favre0[k][j][i] - u_2d0[k][i];
                u_Favre0[k][j][i] = Vx_0[k][j][i] - u_2d0[k][i];

            }

        }

    }



    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                uv_Favre[k][j][i] = v_Favre0[k][j][i] * u_Favre0[k][j][i] * d0[k][j][i];
                uu_Favre[k][j][i] = u_Favre0[k][j][i] * u_Favre0[k][j][i] * d0[k][j][i];

                u_Favre[k][j][i] = Vx_0[k][j][i] * d0[k][j][i];
                v_Favre[k][j][i] = Vr_0[k][j][i] * d0[k][j][i];

            }

        }

    }

    average_data_xz(uv_2d, uv_Favre);
    average_data_xz(uu_2d, uu_Favre);

    average_data_xz(u_2d, u_Favre);
    average_data_xz(v_2d, v_Favre);
    
    

    if(my_id == 0){
        for(int j = 0; j < nz; j++){
          for(int i = 0; i < nx; i++){
              u_2d[j][i] = u_2d[j][i] / d2d[j][i]; 
              v_2d[j][i] = v_2d[j][i] / d2d[j][i];    

            }

        }
        for(int j = 1; j < nz-1; j++){
                for(int i = 1; i < nx-1; i++){

                    uux[j][i] = (uu_2d[j][i+1] - uu_2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                    uy[j][i] = (u_2d[j+1][i] - u_2d[j][i])/(zh2d[j+1][i] - zh2d[j][i]);

                    ux[j][i] = (u_2d[j][i+1] - u_2d[j][i])/(xh2d[j][i+1] - xh2d[j][i]);

                    // uux[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * uu_2d[j][i] - xh2d[j][i] * xh2d[j][i] * uu_2d[j][i+1])/(
                    //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                    // uy[j][i] = (zh2d[j+1][i] * zh2d[j+1][i] * u_2d[j][i] - zh2d[j][i] * zh2d[j][i] * u_2d[j+1][i])/(
                    //             zh2d[j+1][i] * zh2d[j+1][i] * zh2d[j][i] - zh2d[j][i] * zh2d[j][i] * zh2d[j+1][i]);

                    // uy[j][i] = (zh2d[j][i]-zh2d[j-1][i])/((zh2d[j+1][i]-zh2d[j][i])*(zh2d[j+1][i]-zh2d[j-1][i]))*u_2d[j+1][i] + 
                    //            (zh2d[j+1][i]-zh2d[j-1][i])/((zh2d[j+1][i]-zh2d[j][i])*(zh2d[j][i]-zh2d[j-1][i]))*u_2d[j][i] - 
                    //            (zh2d[j+1][i]-zh2d[j][i])/((zh2d[j][i]-zh2d[j-1][i])*(zh2d[j+1][i]-zh2d[j-1][i]))*u_2d[j-1][i];

                    // ux[j][i] = (xh2d[j][i+1] * xh2d[j][i+1] * u_2d[j][i] - xh2d[j][i] * xh2d[j][i] * u_2d[j][i+1])/(
                    //             xh2d[j][i+1] * xh2d[j][i+1] * xh2d[j][i] - xh2d[j][i] * xh2d[j][i] * xh2d[j][i+1]);

                    
                    
                }  
            }
        
        for(int i = 1; i < nx-1; i++){
            for(int j = 2; j < 1.3*boundary_k[i]; j++){
            
                
                // pCfT_in[i] = pCfT_in[i] + (uv_2d[j][i] + uv_2d[j-1][i])* (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD4_in[i] = pCfD4_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * uux[j][i] +
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) * uux[j-1][i])* (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfM_in[i] = pCfM_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * d2d[j][i] * v_2d[j][i] * uy[j][i] +
                //                            (zh2d[nz-1][i] - zh2d[j-1][i]) * d2d[j-1][i] * v_2d[j-1][i] * uy[j-1][i]) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                // pCfD1_in[i] = pCfD1_in[i] + ((zh2d[nz-1][i] - zh2d[j][i]) * d2d[j][i] * u_2d[j][i] * ux[j][i] +
                //                              (zh2d[nz-1][i] - zh2d[j-1][i]) * d2d[j-1][i] * u_2d[j-1][i] * ux[j-1][i]) * (zh2d[j][i] - zh2d[j-1][i])/2.0 ;

                pCfT_in[i] = pCfT_in[i] + uv_2d[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD4_in[i] = pCfD4_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * uux[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfM_in[i] = pCfM_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * d2d[j][i] * v_2d[j][i] * uy[j][i] * (zh2d[j][i] - zh2d[j-1][i]);

                pCfD1_in[i] = pCfD1_in[i] + (zh2d[(int)(1.3*boundary_k[i])][i] - zh2d[j][i]) * d2d[j][i] * u_2d[j][i] * ux[j][i] * (zh2d[j][i] - zh2d[j-1][i]);
            }
        }
        

            for(int i = 1; i < nx-1; i++){

                pCfT[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfT_in[i];

                pCfD4[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD4_in[i]; 

                pCfM[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfM_in[i];  

                pCfD1[i] = (-2.0/(d2d[boundary_k[i]][i] * us2d[boundary_k[i]][i]*us2d[boundary_k[i]][i] * zh2d[(int)(1.3*boundary_k[i])][i])) * pCfD1_in[i];
            }

        printf("Compute CfT CfD4 CfM OK\n");

    }


}



void RD_decomposition(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd0);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu0);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv0);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw0);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT0);

    double (*d2d)[nx] = (double(*)[nx])(pd02d);
    double (*T2d)[nx] = (double(*)[nx])(pT02d);
    double (*p2d)[nx] = (double(*)[nx])(pP02d);
    double (*u2d)[nx] = (double(*)[nx])(pu02d);
    double (*v2d)[nx] = (double(*)[nx])(pv02d);
    double (*w2d)[nx] = (double(*)[nx])(pw02d);

    double (*zh2d)[nx] = (double(*)[nx])(pzh2d);
    double (*xh2d)[nx] = (double(*)[nx])(pxh2d);
    double (*us2d)[nx] = (double(*)[nx])(pus2d);
    double (*uv2d)[nx] = (double(*)[nx])(puv2d);
    double (*uw2d)[nx] = (double(*)[nx])(puw2d);

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
