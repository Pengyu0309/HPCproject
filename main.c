static char help[] = "Sove one-dimensional transient problem\n\n";

#include "petscksp.h"
#include <petscvec.h>
#include "petscmat.h"

int main(int argc, char **argv){
	MPI_Comm        comm;
        Vec             x, t,uu,uu_new,f;
        Mat             A,U;
	KSP             ksp;
	PC              pc;
	PetscInt        ii=0,jj=0,N=10,N1,Nh,Na,n=10;
        PetscScalar     deltax=0.1,deltat=0.1,L=1,T=1,density = 1.0, c = 1.0,l=1.0,k=1.0,h=1.0,g=0;
	PetscInt        implict = 1,heat_flux = 0;
	PetscScalar     temp,print;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
        comm = PETSC_COMM_WORLD;

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-space_num", &N, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-time_num", &n, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num1", &ii, PETSC_NULL);
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num2", &jj, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-implict", &implict, PETSC_NULL);//explict=0;implict=1
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-heat_flux", &heat_flux, PETSC_NULL);//constTEM=0;heatflux=1
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-length", &L, PETSC_NULL);
        PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-time", &T, PETSC_NULL);
        PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-density", &density, PETSC_NULL);//density kg/m^3
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat_capacity", &c, PETSC_NULL);//heat capacity,J/kg-k
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat_supply", &l, PETSC_NULL);//heat supply per unit volume
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-conductivity", &k, PETSC_NULL);//conductivity, W/m-k
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-heat flux", &h, PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-boundary_temperature", &g, PETSC_NULL);//temperature
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-delta_x", &deltax, PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-delta_t", &deltat, PETSC_NULL);
	
	N1 = N-1;
	Nh = N/2;
	Na = N+1;
	//generate vector	
	//space
	VecCreate(comm, &x);
        VecSetSizes(x, PETSC_DECIDE, Na);
        VecSetFromOptions(x);
	VecSet(x, 1.0);
	PetscInt   istart,iend;
        VecGetOwnershipRange(x,&istart,&iend);
        for (ii=istart; ii<iend; ii++) {
                temp = (PetscScalar)(ii*deltax);
                VecSetValues(x,1,&ii,&temp,INSERT_VALUES);
        }
        VecAssemblyBegin(x);
        VecAssemblyEnd(x);
	//VecView(x,PETSC_VIEWER_STDOUT_WORLD);

	//time
	VecCreate(comm, &t);
        VecSetSizes(t, PETSC_DECIDE, n+1);
        VecSetFromOptions(t);
	VecSet(t, 0.0);
        VecGetOwnershipRange(t,&istart,&iend);
        for (ii=istart; ii<iend; ii++) {
                temp = (PetscScalar)(ii*deltat);
                VecSetValues(t,1,&ii,&temp,INSERT_VALUES);
        }
        VecAssemblyBegin(t);
        VecAssemblyEnd(t);
        //VecView(t,PETSC_VIEWER_STDOUT_WORLD);


	//heat supply
        VecCreate(comm, &f);
        VecSetSizes(f, PETSC_DECIDE, Na);
        VecSetFromOptions(f);
        VecSet(f, 0.0);
        VecGetOwnershipRange(f,&istart,&iend);
        for (ii=istart; ii<iend; ii++) {
                temp = (PetscScalar)(sin(l*3.1415*ii*deltax));
                VecSetValues(f,1,&ii,&temp,INSERT_VALUES);
        }
        VecAssemblyBegin(f);
        VecAssemblyEnd(f);
        //VecView(f,PETSC_VIEWER_STDOUT_WORLD);


	//generate matrix A
	MatCreate(comm, &A);
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Na, Na);
	MatSetType(A,MATMPIAIJ);
        MatSetFromOptions(A);
        MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
        MatSeqAIJSetPreallocation(A, 3, NULL);

        PetscInt        rstart, rend, M, m;
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
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	if(heat_flux==1){
	MatSetValue(A,0,0,100,INSERT_VALUES);
	MatSetValue(A,0,1,0,INSERT_VALUES);
	MatSetValue(A,N,N,1,INSERT_VALUES);
	MatSetValue(A,N,N-1,-1,INSERT_VALUES);
	}
	if(heat_flux==0){
        MatSetValue(A,0,0,100,INSERT_VALUES);
        MatSetValue(A,0,1,0,INSERT_VALUES);
        MatSetValue(A,N,N,1,INSERT_VALUES);
        MatSetValue(A,N,N-1,0,INSERT_VALUES);
        }
        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        //MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);


	//generate vector uu uu_new
	VecCreate       (comm,&uu);
	VecCreate       (comm,&uu_new);
        VecSetSizes(uu, PETSC_DECIDE, Na);
        VecSetFromOptions(uu);
        VecDuplicate(uu, &uu_new);
	VecSet(uu, 0.0);
	//set the initial values at t=0
	temp = 0.0;
	VecGetOwnershipRange(uu,&istart,&iend);
        for (ii=istart; ii<iend; ii++) {
                if (ii == 0) {
                        VecSetValues(uu,1,&ii,&g,INSERT_VALUES);
                }
                else{
                        temp = exp(ii*deltax);
                        VecSetValues(uu,1,&ii,&temp,INSERT_VALUES);
                }

        }
        VecAssemblyBegin(uu);
        VecAssemblyEnd(uu);
        if (heat_flux==0){
                temp = 0;
                VecSetValues(uu,1,&N,&temp,INSERT_VALUES);
        }
	if (heat_flux==1){
                VecGetValues(uu,1,&N1,&temp);// heat flux
                temp = temp + h*deltax/k;
                VecSetValues(uu,1,&N,&temp,INSERT_VALUES);
        }
        VecAssemblyBegin(uu);
        VecAssemblyEnd(uu);
        //VecView(uu,PETSC_VIEWER_STDOUT_WORLD);
        VecSet(uu_new, 0.0);
        VecAssemblyBegin(uu_new);
        VecAssemblyEnd(uu_new);
	
	//KSP set
	KSPCreate(comm, &ksp);
	KSPSetType(ksp,KSPCG);
	KSPSetFromOptions(ksp);
	KSPGetPC(ksp, &pc);
        PCSetType(pc, PCJACOBI);
        KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetOperators(ksp, A, A);

	//solve core
	PetscInt       its = 0;
	//Explict
	if (implict == 0){
		PetscScalar      value3,value4,temp1,temp2;
		value3 = deltat/density/c;
		VecScale(f,value3);
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
			VecGetValues(f,1,&ii,&value3);
			temp = temp+value3+value4*(temp1-2*temp+temp2);
			VecSetValue(uu_new,ii,temp,INSERT_VALUES);
		}
		if(heat_flux==1){
                	VecGetValues(uu_new,1,&N1,&temp);
			temp = temp+h*deltax/k;
			VecSetValue(uu_new,N,temp,INSERT_VALUES);
		}
		if(heat_flux==0){
			VecSetValue(uu_new,N,0,INSERT_VALUES);
		}
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
	if (implict == 1){
		PetscScalar    value1,value2,temp3,temp4;
        	value1 = h*deltax/k;
        	value2 = deltat/density/c;
		VecScale(f,value2);
		temp3 = 0;
		temp4 = 0;
	while(its<n){
		if(dt/dx>=0.5){
                        PetscPrintf(comm,"\nThe dx and dt cannot give a satisfied solution.\n";
			break;
                }
                else{
		VecAXPY(uu,1,f);//transfer f from left to right
		VecSetValues(uu,1,&N,&value1,INSERT_VALUES);
		temp = 100*g;
                val = 0;
		VecSetValues(uu,1,&val,&temp,INSERT_VALUES);
                if(heat_flux==0){
                        temp = 0;
                        VecSetValues(uu,1,&N,&temp,INSERT_VALUES);
                }
                VecAssemblyBegin(uu);
                VecAssemblyEnd(uu);
		KSPSolve(ksp,uu,uu_new);
		PetscPrintf(comm,"\nAt time %g:\n",(double)((its+1)*deltat));
                VecView(uu_new,PETSC_VIEWER_STDOUT_WORLD);
		VecCopy(uu_new,uu);
		its++;
	}
	}
	}

	PetscPrintf(comm,"\n\n");

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
