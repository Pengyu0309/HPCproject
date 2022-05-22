#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "petscksp.h"
#include <petscvec.h>
#include "petscmat.h"

#define FLOAT_TYPE double

int main(int argc, char **argv){
	MPI_Comm        comm;
        Vec             x, t;
        Mat             A,U;
	Petscint        i=0,j=0,N=10,n=10;
        PetscReal       L = 1.0,t = 1.0,density = 1.0, c = 1.0,f=1.0,k=1.0,h=1.0,g=10;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
        comm = PETSC_COMM_WORLD;

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-space_num", &N, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-time_num", &n, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num1", &i, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num2", &j, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-length", &L, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time", &t, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-density", &density, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-heat_capacity", &c, PETSC_NULL);//heat capacity,J/kg-k
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-heat_supply", &f, PETSC_NULL);//heat supply per unit volume
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-conductivity", &k, PETSC_NULL);//conductivity, W/m-k
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-heat flux", &h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-boundary_temperature", &g, PETSC_NULL);//temperature
	
	//generate vector	
	//space
	VecCreate(comm, &x);
        VecSetSizes(x, PETSC_DECIDE, L/N+1);
        VecSetFromOptions(x);
	for (i<(N+1);i++){
		VecSetValue(x, i, i*(L/N), INSERT_VALUES);
	}

	//time
	VecCreate(comm, &t);
        VecSetSizes(t, PETSC_DECIDE, t/n+1);
        VecSetFromOptions(t);
        for (i<(n+1);i++){
                VecSetValue(t, i, i*(t/n), INSERT_VALUES);
        }

	//generate matrix A
	MatCreate(comm, &A);
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, L/N+1, L/N+1);
        MatSetFromOptions(A);
        MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
        MatSeqAIJSetPreallocation(A, 3, NULL);

        PetscInt        rstart, rend, M, m;
        MatGetOwnershipRange(A, &rstart, &rend);
        MatGetSize(A, &M, &m);
        for (PetscInt i=rstart; i<rend; i++) {
                PetscInt    index[3] = {i-1, i, i+1};
                PetscScalar value[3] = {-1.0, 2.0, -1.0};
                if (i == 0) {
                        MatSetValues(A, 1, &i, 2, &index[1], &value[1], INSERT_VALUES);
                }
                else if (i == m-1) {
                        MatSetValues(A, 1, &i, 2, index, value, INSERT_VALUES);
                }
                else {
                        MatSetValues(A, 1, &i, 3, index, value, INSERT_VALUES);
                }
        }
	MatSetValue(A,0,0,1,INSERT_VALUES);
	MatSetValue(A,m-1,m-2,0,INSERT_VALUES);
	MatSetValue(A,m-1,m-1,1,INSERT_VALUES);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

//      MatView(A, PETSC_VIEWER_STDOUT_WORLD);

        MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);









//print the values of x	
//	printf("x[%d] is: \n",N+1);
//		for(i=0;i<N+1;i++){
//			printf("%f\t",x[i]);}

//print the values of t
//	printf("t[%d] is: \n",M);
//                for(i=0;i<M;i++){
//                        printf("%f\t",t[i]);}


        //critical time determination
	alpha = 1/(density*c);
	double delta_t_critical;
	delta_t_critical = ((1/alpha)*(dl*2)/(2*k));

	
        //initial condition
	double u0=10.0;
	FLOAT_TYPE *u;//temperature
        u = (FLOAT_TYPE*) malloc(sizeof(double)*M*N);
	for (i=0;i<M;i++){
		for (j=0;j<N;j++){
			if (i==0){
		            u[j]=u0;}else{
			u[N*i+j]=0;}
		}

	}
//print the values of u
//        printf("u[%d][%d] is: \n",M,N);
//	for (i=0;i<M;++i)
//	{
//		for (j=0;j<N;++j)
//		{
//			printf("%f\t",u[i*N+j]);
//		}
//		printf("\n");
//	}
	

}
