static char help[] = "Hello world!\n\n";
#include <stdio.h>
#include <stdlib.h>
#include "petscksp.h"
#include <petscvec.h>
#include "petscmat.h"

int main(int argc, char **argv){
	MPI_Comm        comm;
        Vec             x, t;
        Mat             A,U;
	KSP             ksp;
	PC              pc;
	PetscInt        ii=0,jj=0,N=10,n=10;
        PetscScalar     deltax=0.1,deltat=0.1,L=1,T=1,density = 1.0, c = 1.0,f=1.0,k=1.0,h=1.0,g=10;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
        comm = PETSC_COMM_WORLD;

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-space_num", &N, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-time_num", &n, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num1", &ii, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num2", &jj, PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-length", &L, PETSC_NULL);
        PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-time", &T, PETSC_NULL);
        PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-density", &density, PETSC_NULL);//density kg/m^3
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat_capacity", &c, PETSC_NULL);//heat capacity,J/kg-k
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat_supply", &f, PETSC_NULL);//heat supply per unit volume
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-conductivity", &k, PETSC_NULL);//conductivity, W/m-k
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat flux", &h, PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-boundary_temperature", &g, PETSC_NULL);//temperature
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-delta_x", &deltax, PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-delta_t", &deltat, PETSC_NULL);
	
	PetscScalar    print;
	//generate vector	
	//space
	VecCreate(comm, &x);
        VecSetSizes(x, PETSC_DECIDE, N+1);
        VecSetFromOptions(x);
	VecSet(x, 0.0);
	for (ii=0;ii<(N+1);ii++){
		VecSetValue(x, ii, ii*L/N, INSERT_VALUES);
	}
	VecAssemblyBegin(x);
        VecAssemblyEnd(x);
	//verify the grid x
	PetscPrintf(comm,"\nThe Vector x is: ");
	for (ii=0;ii<(N+1);ii++){
                VecGetValues(x,1,&ii,&print);
		PetscPrintf(comm,"%g\t",print);
        }

	//time
	VecCreate(comm, &t);
        VecSetSizes(t, PETSC_DECIDE, n+1);
        VecSetFromOptions(t);
	VecSet(t, 0.0);
        for (ii=0;ii<(n+1);ii++){
                VecSetValue(t, ii, ii*T/n, INSERT_VALUES);
        }
	VecAssemblyBegin(t);
        VecAssemblyEnd(t);
	//verify the grid t
        PetscPrintf(comm,"\nThe Vector t is: ");
	for (ii=0;ii<(n+1);ii++){
                VecGetValues(t,1,&ii,&print);
                PetscPrintf(comm,"%g\t",print);
        }

	//generate matrix A
	MatCreate(comm, &A);
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N+1, N+1);
        MatSetFromOptions(A);
        MatMPIAIJSetPreallocation(A, 3, NULL, 8, NULL);
        MatSeqAIJSetPreallocation(A, 3, NULL);

        PetscInt        rstart, rend, M, m;
	PetscScalar     kk;
	kk = k*T/n/density/c/(L/N)/(L/N);
        MatGetOwnershipRange(A, &rstart, &rend);
        MatGetSize(A, &M, &m);
        for (PetscInt i=rstart; i<rend; i++) {
                PetscInt    index[3] = {i-1, i, i+1};
                PetscScalar value[3] = {(-k*deltat/density/c/deltax/deltax), (1+2*k*deltat/density/c/deltax/deltax), (-k*deltat/density/c/deltax/deltax)};
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
	MatSetValue(A,0,1,0,INSERT_VALUES);
	MatSetValue(A,m-1,m-1,1,INSERT_VALUES);
	MatSetValue(A,m-1,m-2,-1,INSERT_VALUES);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

        MatView(A, PETSC_VIEWER_STDOUT_WORLD);

        MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);

	//generate matrix U
        MatCreate(comm, &U);
        MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, n+1, N+1);
        MatSetFromOptions(U);
        MatMPIAIJSetPreallocation(U, 3, NULL, 3, NULL);
        MatSeqAIJSetPreallocation(U, 3, NULL);

	PetscInt        rstart1, rend1, M1, m1,N1;
	N1 = N-1;
	PetscScalar       temp;
	temp = 0.0;
        MatGetOwnershipRange(U, &rstart1, &rend1);
        MatGetSize(U, &M1, &m1);
	for (PetscInt i1=rstart1; i1<rend1; i1++) {
                if (i1 == 0) {
                        MatSetValue(U,i1,0,g,INSERT_VALUES);
			for (PetscInt j1=1; j1<N; j1++){
				MatSetValue(U,i1,j1,1,INSERT_VALUES);
			}
			MatGetValues(U,1,&i1,1,&N1,&temp);
			MatSetValue(U,i1,N,(temp+h*deltax/k), INSERT_VALUES);//initial temperature
                }
                else {
                        MatSetValue(U,i1,0, g, INSERT_VALUES);
                }
        }
	MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);

        MatView(U, PETSC_VIEWER_STDOUT_WORLD);
        MatSetOption(U, MAT_SYMMETRIC, PETSC_TRUE);
	

	//KSP
	KSPCreate(comm, &ksp);
	KSPSetType(ksp,KSPCG);
	KSPSetFromOptions(ksp);
	KSPGetPC(ksp, &pc);
        PCSetType(pc, PCJACOBI);
        KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetOperators(ksp, A, A);

	//solve implict
	PetscInt       its = 0;
	PetscScalar    value1;
	value1 = h*deltax/k;
	Vec       uu,uu_new;
	VecSetSizes(uu, PETSC_DECIDE, N+1);
        VecSetFromOptions(uu);
	VecDuplicate(uu, &uu_new);
	VecSet(uu, 0.0);
	VecAssemblyBegin(uu);
        VecAssemblyEnd(uu);
	VecSet(uu_new, 0.0);
        VecAssemblyBegin(uu_new);
        VecAssemblyEnd(uu_new);
	//VecView(znew, PETSC_VIEWER_STDOUT_WORLD);

	while(its<n){
		for(ii=0;ii<N+1;ii++){
			MatGetValues(U,1,&its,1,&ii,&temp);
			VecSetValues(uu,1,&ii,&temp, INSERT_VALUES);
		}
		VecSetValues(uu,1,&N,&(value1), INSERT_VALUES);
		VecShift(uu,f*T/n/density/c);//transfer f from left to right
		KSPSolve(ksp,uu,uu_new);
		for(jj=0;jj<N+1;jj++){
                        VecGetValues(uu_new,1,&jj,&temp);
                        MatSetValues(U,1,&its+1,1,&jj,&temp,INSERT_VALUES);
                }
	}
	MatView(U, PETSC_VIEWER_STDOUT_WORLD);

	//destroy
	KSPDestroy(&ksp);
	VecDestroy(&x);
	VecDestroy(&t);
	VecDestroy(&uu);
	VecDestroy(&uu_new);
	MatDestroy(&A);
	MatDestroy(&U);

	PetscFinalize();
        return 0;
}