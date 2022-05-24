static char help[] = "Hello world!\n\n";
#include <stdio.h>
#include <stdlib.h>
#include "petscksp.h"
#include <petscvec.h>
#include "petscmat.h"

int main(int argc, char **argv){
	MPI_Comm        comm;
        Vec             x, t,uu,uu_new;
        Mat             A,U;
	KSP             ksp;
	PC              pc;
	PetscInt        ii=0,jj=0,N=10,n=10;
        PetscScalar     deltax=0.1,deltat=0.1,L=1,T=1,density = 1.0, c = 1.0,f=1.0,k=1.0,h=1.0,g=10;
	PetscInt        choice = 1;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
        comm = PETSC_COMM_WORLD;

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-space_num", &N, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-time_num", &n, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num1", &ii, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num2", &jj, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-choice", &choice, PETSC_NULL);//explict=0;implict=1
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
	//VecSetType(x,VECMPI);
        VecSetFromOptions(x);
	VecSet(x, 0.0);
	for (ii=0;ii<(N+1);ii++){
		VecSetValue(x, ii, ii*deltax, INSERT_VALUES);
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
	//VecSetType(t,VECMPI);
        VecSetFromOptions(t);
	VecSet(t, 0.0);
        for (ii=0;ii<(n+1);ii++){
                VecSetValue(t, ii, ii*deltat, INSERT_VALUES);
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
	MatSetType(A,MATMPIAIJ);
        MatSetFromOptions(A);
        MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
        MatSeqAIJSetPreallocation(A, 3, NULL);

        PetscInt        rstart, rend, M, m;
	PetscScalar     kk;
	kk = k*deltat/density/c/deltax/deltax;
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
	MatSetValue(A,0,0,100,INSERT_VALUES);
	MatSetValue(A,0,1,0,INSERT_VALUES);
	MatSetValue(A,N,N,1,INSERT_VALUES);
	MatSetValue(A,N,N-1,-1,INSERT_VALUES);

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

        MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);


	//generate vector uu uu_new
	VecCreate       (comm,&uu);
	VecCreate       (comm,&uu_new);
        VecSetSizes(uu, PETSC_DECIDE, (N+1));
	//VecSetType(uu,VECMPI);
        VecSetFromOptions(uu);
        VecDuplicate(uu, &uu_new);
	VecSetType(uu_new,VECMPI);
	VecSet(uu, 0.0);
	//set the initial values at t=0
	PetscInt          N1;
	N1 = N-1;
	PetscScalar       temp;
	temp = 0.0;
	for (ii=0; ii<(N+1); ii++) {
                if (ii == 0) {
                        VecSetValue(uu,ii,g/100,INSERT_VALUES);
		}
		else if (ii<N){
			temp = exp(ii*deltax);
			VecSetValue(uu,ii,temp,INSERT_VALUES);
		}
                else {
                        VecGetValues(uu,1,&N1,&temp);// heat flux
			temp = temp + h*deltax/k;
			VecSetValue(uu,ii,temp,INSERT_VALUES);
                }
        }

	VecAssemblyBegin(uu);
        VecAssemblyEnd(uu);
	//verify the vector uu
	//PetscPrintf(comm,"\nThe Vector uu is: ");
        //for (ii=0;ii<(N+1);ii++){
	//	if(ii=0){
	//		PetscPrintf(comm,"%g\t",g);
	//	}else{
        //        VecGetValues(uu,1,&ii,&print);
        //        PetscPrintf(comm,"%g\t",print);
        //}
	//}
        VecSet(uu_new, 0.0);
        VecAssemblyBegin(uu_new);
        VecAssemblyEnd(uu_new);
	

	//KSP
	KSPCreate(comm, &ksp);
	KSPSetType(ksp,KSPCG);
	KSPSetFromOptions(ksp);
	KSPGetPC(ksp, &pc);
        PCSetType(pc, PCJACOBI);
        KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetOperators(ksp, A, A);

	//solve
	PetscInt       its = 0;
	//Explict
	if (choice == 0){
		PetscScalar      value3,value4,temp1,temp2;
		value3 = f*deltat/density/c;
		value4 = k*deltat/density/c/deltax/deltax;
		temp1 = 0.0;
		temp2 = 0.0;
        while(its<n){
                VecSetValue(uu_new,0,g,INSERT_VALUES);
		for (ii=1;ii<N;ii++){
			PetscInt  value5, value6;
			value5 = ii-1;
			value6 = ii+1;
			VecGetValues(uu,1,&ii,&temp);
			VecGetValues(uu,1,&value5,&temp1);
			VecGetValues(uu,1,&value6,&temp2);
			temp = temp+value3+value4*(temp1-2*temp+temp2);
			VecSetValue(uu_new,ii,temp,INSERT_VALUES);
		}
                VecGetValues(uu_new,1,&N1,&temp);
		temp = temp+h*deltax/k;
		VecSetValue(uu_new,N,temp,INSERT_VALUES);
                PetscPrintf(comm,"\nAt time %g, the temperature distribution is\n",(double)((its+1)*deltat));
                for(jj=0;jj<N+1;jj++){
                        VecGetValues(uu_new,1,&jj,&print);
                        PetscPrintf(comm,"%g\t",print);
                }
		VecCopy(uu_new,uu);
		its++;
        }
        }

	//solve implict
	if (choice == 1){
		PetscScalar    value1,value2;
        	value1 = h*deltax/k;
        	value2 = f*deltat/density/c;
	while(its<n){
		VecShift(uu,value2);//transfer f from left to right
		VecSetValues(uu,1,&N,&value1,INSERT_VALUES);
		VecSetValue(uu,0,g/100,INSERT_VALUES);
		//PetscPrintf(comm,"\nvector u is\n");
                //for(jj=0;jj<N+1;jj++){
                //        VecGetValues(uu,1,&jj,&print);
                //        PetscPrintf(comm,"%g\t",print);
		//}
		KSPSolve(ksp,uu,uu_new);
		PetscPrintf(comm,"\nAt time %g, the temperature distribution is\n %g\t",(double)((its+1)*deltat),g);
		for(jj=1;jj<N+1;jj++){
			VecGetValues(uu_new,1,&jj,&print);
                        PetscPrintf(comm,"%g\t",print);
		}
		VecCopy(uu_new,uu);
		its++;
	}
	}

	//destroy
	KSPDestroy(&ksp);
	VecDestroy(&x);
	VecDestroy(&t);
	VecDestroy(&uu);
	VecDestroy(&uu_new);
	MatDestroy(&A);

	PetscFinalize();
        return 0;
}
