#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LinearSparse.c"
#else

void THNN_(LinearSparse_updateOutput)(
           THNNState * state,
           THTensor * inputT,
           THTensor * outputT,
           THTensor * weightT,
           THIntegerTensor * rowsT,
           THIntegerTensor * colsT,
           THIntegerTensor * iRowStart,
           int nRowWeight,
           int nColWeight)
                
{
    int i, j, k;
  real *input; 
  real *weight; 
  real *output;
  int * rows;
  int * cols;
  int nnz;
  int nRowInp, nColInp;

   nRowInp = inputT->size[0];
   nColInp = inputT->size[1];
   nnz = weightT->size[0];


   input = THTensor_(data)(inputT);
   weight = THTensor_(data)(weightT);
   output = THTensor_(data)(outputT);
   rows = THIntegerTensor_(data)(rowsT);
   cols = THIntegerTensor_(data)(colsT);


   //printf("Input Size: %ld, %ld ... stride: %ld, %ld\n", inputT->size[0] ,inputT->size[1], inputT->stride[0], inputT->stride[1]);
   //printf("Output Size: %ld, %ld ... stride: %ld, %ld\n", outputT->size[0] ,outputT->size[1], outputT->stride[0], outputT->stride[1]);

/*printf("input: \n");
   for(i=0; i<nRowInp; i++)
   {
     for(j=0; j<nColInp; j++)
        printf("%f, ", input[nColInp*i + j]);
     printf("\n");
   }

printf("Pesos: \n");

   for(i=0;i<nnz;i++)
   {
      printf("%f, ", weight[i]);
   }

printf("Indices: \n");

for(i=0;i<nnz;i++)
   {
    printf("(%d,%d), ", rows[i], cols[i]);
   }*/

   if(inputT->nDimension == 1)
   {

	   k=0;
	   for(i=0; i<nRowWeight; i++)
	   {
	      
		  double sum = 0.0;

		  while(k < nnz && rows[k] == i)
		  {
		     sum += input[cols[k]] * weight[k];
		     k++;
		  }

		  output[i] += sum;
	    }

   }
   else if(inputT->nDimension == 2)
   {
#pragma parallel for
	       for(i=0; i<nRowInp; i++)
	       {
		 k=0;

	      for(j=0; j < nRowWeight; j++)
	      {
		  double sum = 0.0;

		  while(k < nnz && rows[k] == j)
		  {
		     sum += input[i*inputT->stride[0] + cols[k]*inputT->stride[1]] * weight[k];
		     k++;
		  }

		  output[i*outputT->stride[0] + j*outputT->stride[1]] = sum;
	      }
	    }


   }
}

void THNN_(LinearSparse_updateGradInput)(
           THNNState * state,
           THTensor * inputDummy,
           THTensor * inputT,
           THTensor * outputT,
           THTensor * weightT,
           THIntegerTensor * rowsT,
           THIntegerTensor * colsT,
           int nRowWeight,
           int nColWeight,
           THIntegerTensor * iRowStartT)
{
    int i, j, k;
  real *input; 
  real *weight; 
  real *output;
  int * rows;
  int * cols;
  int * iRowStart;
  int nnz;
  int nRowInp, nColInp;

   nRowInp = inputT->size[0];
   nColInp = inputT->size[1];
   nnz = weightT->size[0];


   input = THTensor_(data)(inputT);
   weight = THTensor_(data)(weightT);
   output = THTensor_(data)(outputT);
   rows = THIntegerTensor_(data)(rowsT);
   cols = THIntegerTensor_(data)(colsT);
   iRowStart = THIntegerTensor_(data)(iRowStartT);


   //printf("Input Size: %ld, %ld ... stride: %ld, %ld\n", inputT->size[0] ,inputT->size[1], inputT->stride[0], inputT->stride[1]);
   //printf("Output Size: %ld, %ld ... stride: %ld, %ld\n", outputT->size[0] ,outputT->size[1], outputT->stride[0], outputT->stride[1]);

/*printf("input: \n");
   for(i=0; i<nRowInp; i++)
   {
     for(j=0; j<nColInp; j++)
        printf("%f, ", input[nColInp*i + j]);
     printf("\n");
   }

printf("Pesos: \n");

   for(i=0;i<nnz;i++)
   {
      printf("%f, ", weight[i]);
   }

printf("Indices: \n");

for(i=0;i<nnz;i++)
   {
    printf("(%d,%d), ", rows[i], cols[i]);
   }*/

  int actWeightRow = 0;
  int iWeight = 0;
  double sum = 0.0;


      if(outputT->nDimension == 1)
   {

	      for(j=0; j < nColWeight; j++)
	      {
		  sum = 0.0;

		  actWeightRow = 0;
		  iWeight = iRowStart[actWeightRow] + j;

		  while(cols[iWeight] == j)
		  {
		     sum += input[rows[iWeight]] * weight[iWeight];
		     //printf("index inp: %d  = %d + %d",i*inputT->stride[0] + rows[iWeight]*inputT->stride[1], i*inputT->stride[0], rows[iWeight]*inputT->stride[1]);
		     //printf("\ni: %d, j: %d, k: %d ... sum: %f = i:%f x w:%f\n", i, j, k, sum, input[i*inputT->stride[0] + rows[iWeight]*inputT->stride[1]], weight[iWeight]);
		     actWeightRow++;
		     if(actWeightRow >= nRowWeight)
		         break;
		     iWeight = iRowStart[actWeightRow] + j;
		  }

		  output[j] = sum;
	      }
	    
   }
   else if(outputT->nDimension == 2)
   {
        #pragma omp parallel for
	for(i=0; i<nRowInp; i++)
	   {
	      for(j=0; j < nColWeight; j++)
	      {
		  sum = 0.0;

		  actWeightRow = 0;
		  iWeight = iRowStart[actWeightRow] + j;

		  while(cols[iWeight] == j)
		  {
		     sum += input[i*inputT->stride[0] + rows[iWeight]*inputT->stride[1]] * weight[iWeight];
		     //printf("index inp: %d  = %d + %d",i*inputT->stride[0] + rows[iWeight]*inputT->stride[1], i*inputT->stride[0], rows[iWeight]*inputT->stride[1]);
		     //printf("\ni: %d, j: %d, k: %d ... sum: %f = i:%f x w:%f\n", i, j, k, sum, input[i*inputT->stride[0] + rows[iWeight]*inputT->stride[1]], weight[iWeight]);
		     actWeightRow++;
		     if(actWeightRow >= nRowWeight)
		         break;
		     iWeight = iRowStart[actWeightRow] + j;
		  }

		  output[i*outputT->stride[0] + j*outputT->stride[1]] = sum;
	      }
	    }

   }

}



void THNN_(LinearSparse_accGradParameters)(
           THNNState * state,
           THTensor * inputT,
           THTensor * gradOutT,
           THTensor * gradWeightT,
           THIntegerTensor * rowsT,
           THIntegerTensor * colsT,
	   int nnz,
           float scale)
{
  int i, j;
  int * rows;
  int * cols;
  int batchSize;
  int nRowGradOut, nColGradOut, nRowInput, nColInput;
  
  real scale_r = (real) scale;

  real * input, *gradOut, *gradWeight;

   nRowGradOut = gradOutT->size[0];
   nColGradOut = gradOutT->size[1];

   nRowInput = inputT->size[0];
   nColInput = inputT->size[1];

   batchSize = nRowInput;

   input = THTensor_(data)(inputT);
   gradOut = THTensor_(data)(gradOutT);
   gradWeight = THTensor_(data)(gradWeightT);
   rows = THIntegerTensor_(data)(rowsT);
   cols = THIntegerTensor_(data)(colsT);


   /*printf("input: \n");
   for(i=0; i<nRowInput; i++)
   {
     for(j=0; j<nColInput; j++)
        printf("%f, ", input[nColInput*i + j]);
     printf("\n");
   }

   printf("GradOut: \n");
   for(i=0; i<nRowGradOut; i++)
   {
     for(j=0; j<nColGradOut; j++)
        printf("%f, ", gradOut[nColGradOut*i + j]);
     printf("\n");
   }

printf("Indices: \n");

for(i=0;i<nnz;i++)
   {
    printf("(%d,%d), ", rows[i], cols[i]);
   }*/

   if(gradOutT->nDimension == 2)
   {
           #pragma omp parallel for
	   for(i=0; i<nnz; i++)
	   {
	       double sum = 0.0;

	       for(j=0; j<batchSize; j++)
	       {
		   sum += gradOut[j*gradOutT->stride[0] + rows[i]*gradOutT->stride[1]] * input[j*inputT->stride[0] + cols[i]*inputT->stride[1]];
		   //printf("sum %f = %f * %f   -- index 1: %d = %d + %d, index2: %d = %d + %d\n", sum, gradOut[j*gradOutT->stride[0] + rows[i]*gradOutT->stride[1]], input[j*inputT->stride[0] + cols[i]*inputT->stride[1]], j*gradOutT->stride[0] + rows[i]*gradOutT->stride[1], j*gradOutT->stride[0], rows[i]*gradOutT->stride[1], j*inputT->stride[0] + cols[i]*inputT->stride[1], j*inputT->stride[0], cols[i]*inputT->stride[1]);
	       }

	       gradWeight[i] += sum*scale_r;
	   }
   }
   else if(gradOutT->nDimension == 1)
   {
           for(i=0; i<nnz; i++)
	   {
	       gradWeight[i] += gradOut[rows[i]] * input[cols[i]] * scale_r;
	   }
   }
}

#endif

