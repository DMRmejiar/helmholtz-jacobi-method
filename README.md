# Monte Carlo Pi - Secuencial y Paralelo

- David Mejía Restrepo

## Método

Para realizar la paralelización del metodo Jacobi, se utilizo la API de **OpenMP**, buscando la mejor distribucion de los datos entre los hilos para alcanzar el maximo desempeño posible por este metodo.

Para la implementación de la paralelización del codigo en el archivo *code/jacobi.c* se realizo en primera instancia un perfilamiento del codigo, donde se evidencio que la seccion de codigo que requera más capacidad de computo correspondia a la funcion *jacobi* mostrada en la *Figura 1*.
``` c
void jacobi ( int n, int m, double dx, double dy, double alpha, double omega, double *u, double *f, double tol, int maxit, int n_threads )
{
    int i,j,k;
    double error, resid, ax, ay, b;
    double *uold;

    uold = (double *) malloc(sizeof(double)*n*m);
    if (!uold)
    {
        fprintf(stderr, "Error: cant allocate memory\n");
        exit(1);
    }
    ax = 1.0/(dx * dx); /* X-direction coef */
    ay = 1.0/(dy*dy); /* Y_direction coef */
    b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */
    error = 10.0 * tol;
    k = 1;
    while (k <= maxit && error > tol) 
    {
        error = 0.0;
        /* copy new solution into old */
        for (j=0; j<m; j++)
            for (i=0; i<n; i++)
            {
                UOLD(j,i) = U(j,i);
            }
        /* compute stencil, residual and update */
        for (j=1; j<m-1; j++)
        {
            for (i=1; i<n-1; i++)
            {
                resid = (
                    ax * (UOLD(j,i-1) + UOLD(j,i+1))
                    + ay * (UOLD(j-1,i) + UOLD(j+1,i))
                    + b * UOLD(j,i) - F(j,i)
                    ) / b;
                /* update solution */
                U(j,i) = UOLD(j,i) - omega * resid;
                /* accumulate residual error */
                error =error + resid*resid;
            }
        }
        /* error check */
        k++;
        error = sqrt(error) /(n*m);
    } /* while */
    printf("Total Number of Iterations %d\n", k);
    printf("Residual %.15g\n", error);
}
```
*Figura 1. Funcion secuencial jacobi.*

Haciendo pruebas y analizando el problema se realizaron multipes codigos parallelos abordando diferentes posibles soluciones al problema como un paralelismo utilizando *tasks* y asignando una fraccion de los ciclos a cada hilo, en estas se abordo paralelizar las iteraciones realizdas en la función en *while (k <= maxit && error > tol)* y en el ciclo for correspndiente a *compute stencil, residual and update*. Despues de analizar los multiples metodos posibles se opto por no utilizar *tasks* y por paralelizar unicamente el ciclo for correspondiente a *compute stencil, residual and update* dado que es el que presentaba un mejor desempeño. La paralelizacion de esta seccion se muestra en la *figura 2*, y se encuentra en el archivo *code/jacobi-p.c*.

``` c
//Start parallel region
int local_m, local_j, my_rank, size;
size = (m - 2) / n_threads;
#pragma omp parallel \
    num_threads(n_threads) \
    private (i, resid, my_rank, local_j, local_m ) \
    reduction (+:error)
{
    my_rank = omp_get_thread_num();
    local_m = size*(my_rank+1);
    local_j = (size*my_rank)+1;
    
    if (my_rank == (n_threads - 1))
        local_m += (m-2)%n_threads;
    
    /* compute stencil, residual and update */
    for (local_j; local_j<local_m; local_j++)
    {
        for (i=1; i<n-1; i++)
        {
            resid = (
                ax * (UOLD(local_j,i-1) + UOLD(local_j,i+1))
                + ay * (UOLD(local_j-1,i) + UOLD(local_j+1,i))
                + b * UOLD(local_j,i) - F(local_j,i)
                ) / b;
            /* update solution */
            U(local_j,i) = UOLD(local_j,i) - omega * resid;
            /* accumulate residual error */
            error += resid*resid;
        }
    }
}
```
*Figura 2. Fragmento paralelo de la funcion jacobi.*

## Resultados

Para las pruebas se realizaron 30 ejecuciones de la version secuencial del codigo y 30 de la version paralela para 1, 2, 4, 8 y 16 hilos, se utilizo un *grid* de 2000 x 2000, la constante de helmholtz igual a 0.8, una tolerancia al error de 1e-15 y con 50 iteraciones.

El promedio de tiempo de ejecucion y MFlops para la version secuencial se presentan en la fila correspondiente a **seq** de la *Tabla 1*, y adicional en las filas siguientes se presentan para 1, 2, 4, 8 y 16 hilos para la version paralela del codigo, junto con esta informacion se encuentra el Speedup y Efficiency a cada una de estas.

| Threads | Execution time | MFlops | Speedup | Efficiency |
|---|---|---|---|---|---|
| Seq  | 3.005042933 | 864.0252667 | 1 | 1 |
| 1    | 2.817238133 | 921.6523333 | 1.066662735 | 1.0666627350 |
| 2    | 2.329152967 | 1114.674333 | 1.290187023 | 0.6450935117 |
| 4    | 1.854693967 | 1400.151333 | 1.620236539 | 0.4050591347 |
| 8    | 1.498471300 | 1734.784333 | 2.005405731 | 0.2506757164 |
| 16   | 1.436200167 | 1807.450000 | 2.092356625 | 0.1307722891 |

---

Se presentan a continuacion las graficas de speedup y efficiency en la *figura 3* y *figura 4* correspondientemente.

<div align='center'>
    <img src='img/Speedup.png' alt="Diagrama de lineas speedup" width=50%>
    </br><i>Figura 3. Speedup.</i>
    </br></br>
    <img src='img/Efficiency.png' alt="Diagrama de lineas efficiency" width=50%>
    </br><i>Figura 4. Efficiency.</i>
    </br></br>
</div>

## Conclusiones

Se evidencia que a pesar de haber partido de diferentes enfoques para paralelizar el codigo objetivo, el acercamiento tomado puede no ser el más efectivo, dado que se hace evidente que la eficiencia con muliples cores disminuye drasticamente, teniendo eficiancias por debajo de *0.5* a partir de 4 cores.