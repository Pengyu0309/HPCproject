#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#define FLOAT_TYPE double

int main(){
	int i,j;

	double L = 1.0;
	double density = 1.0;
	double c =1;//heat capacity,J/kg-k
	double f=1;//heat supply per unit volume
	double k=1;//conductivity, W/m-k
	double alpha;

	
	//grid
	int N = 10;
        double dl;
	FLOAT_TYPE *x;
	x = (FLOAT_TYPE*) malloc(sizeof(double)*(N+1));
	dl = L/N;
	for (i=0;i<N+1;i++){
		x[i]=i*dl;
	}

//print the values of x	
//	printf("x[%d] is: \n",N+1);
//		for(i=0;i<N+1;i++){
//			printf("%f\t",x[i]);}

	//time
	double time = 1;
	double dt=0.1;
	int M;
	M=(time/dt)+1;
	FLOAT_TYPE *t;
        t = (FLOAT_TYPE*) malloc(sizeof(double)*M);
	for (i=0; i<M;i++){
		t[i]=i*dt;
	}

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
