// kmeans_starter.c

#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define KMEANSITERS 10

// compile
// mpicc kmeans.c -lm -o kmeans

// run example with 2 means
// mpirun -np 4 -hostfile myhostfile.txt ./kmeans 5159737 2 2 iono_57min_5.16Mpts_2D.txt

// function prototypes
int importDataset(char *fname, int DIM, int N, double **dataset);

int main(int argc, char **argv)
{

    int my_rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Process command-line arguments
    int N;
    int DIM;
    int KMEANS;
    char inputFname[500];

    if (argc != 5)
    {
        fprintf(stderr, "Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point/feature vector), K (number of means), dataset filename. Your input: %s\n", argv[0]);
        MPI_Finalize();
        exit(0);
    }

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &DIM);
    sscanf(argv[3], "%d", &KMEANS);
    strcpy(inputFname, argv[4]);

    // pointer to entire dataset
    double **dataset;

    if (N < 1 || DIM < 1 || KMEANS < 1)
    {
        printf("\nOne of the following are invalid: N, DIM, K(MEANS)\n");
        MPI_Finalize();
        exit(0);
    }
    // All ranks import dataset
    else
    {

        if (my_rank == 0)
        {
            printf("\nNumber of lines (N): %d, Dimensionality: %d, KMEANS: %d, Filename: %s\n", N, DIM, KMEANS, inputFname);
        }

        // allocate memory for dataset
        dataset = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++)
        {
            dataset[i] = (double *)malloc(sizeof(double) * DIM);
        }

        int ret = importDataset(inputFname, DIM, N, dataset);

        if (ret == 1)
        {
            MPI_Finalize();
            return 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Write code here

    int i, j, k, l;
    int numPointsForMyRank;

    // Local Time measurement variables
    double tstart_distance, tend_distance, local_time_taken_distance = 0;
    double tstart_updating_1, tend_updating_1, tstart_updating_2, tend_updating_2;    
    double tstart_updating_3, tend_updating_3, local_time_taken_updating = 0;
    double tstart_total, tend_total, local_time_taken_total = 0;

    // Global Time measurement variables
    double global_time_taken_distance;
    double global_time_taken_updating;
    double global_time_taken_total;
    
    // To store the KMEANS number of centroids
    double **centroids;
    centroids = (double **)malloc(sizeof(double *) * (KMEANS));
    for (i = 0; i < KMEANS; i++)
    {
        centroids[i] = (double *)malloc(sizeof(double) * DIM);
    }

    int *numPointsForEachRank;

    if (my_rank == 0)
    {
        numPointsForEachRank = (int *)malloc(sizeof(int) * (nprocs));
    }

    // To find number of points given to each rank:
    int numPointsForRegularRank = N / nprocs;
    int numPointsLeftOver = N % nprocs;

    // Calculate number of points given to each rank at Rank 0:
    if (my_rank == 0)
    {
        for (i = 0; i < nprocs; i++)
        {
            numPointsForEachRank[i] = numPointsForRegularRank;

            // Distribute leftover points over other ranks
            // Ex.: If 5 points leftover, give one extra point to first 5 ranks.
            if (i < numPointsLeftOver)
            {
                numPointsForEachRank[i] = numPointsForEachRank[i] + 1;
            }
        }
    }

    // Send number of points for each rank from rank 0
    MPI_Scatter(numPointsForEachRank, 1, MPI_INT, &numPointsForMyRank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int myStartPoint;

    if (my_rank < numPointsLeftOver)
    {
        myStartPoint = numPointsForMyRank * my_rank;
    }
    else
    {
        myStartPoint = (numPointsForRegularRank * my_rank) + numPointsLeftOver;
    }

    // To store the assigned centroid for each point
    // Store the Array Index of the assigned Centroid instead of values for all dimensions.
    int *assignedCentroid;
    assignedCentroid = (int *)malloc(sizeof(int *) * numPointsForMyRank);

    double distance, minimumDistance;

    // To store the sum of coordinates of the points assigned to a Centroid at each rank
    double **localSumOfCentroidPoints;
    localSumOfCentroidPoints = (double **)malloc(sizeof(double *) * KMEANS);
    for (i = 0; i < KMEANS; i++)
    {
        localSumOfCentroidPoints[i] = (double *)malloc(sizeof(double) * DIM);
    }

    // To store the number of points assigned to a Centroid at each rank
    int *localNumOfPointsForCentroid;
    localNumOfPointsForCentroid = (int *)malloc(sizeof(int) * KMEANS);

    // To store the Global Sum of coordinates of the points assigned to a Centroid 
    double **globalSumOfCentroidPoints;
    globalSumOfCentroidPoints = (double **)malloc(sizeof(double *) * KMEANS);
    for (i = 0; i < KMEANS; i++)
    {
        globalSumOfCentroidPoints[i] = (double *)malloc(sizeof(double) * DIM);
    }


    // To store the number of points assigned to a Centroid globally
    int *globalNumOfPointsForCentroid;
    globalNumOfPointsForCentroid = (int *)malloc(sizeof(int) * KMEANS);

    // For validation purposes:
    double maxCentroid, minCentroid;
    int flag = 0;

    // Assign initial values for the k centroids as the first KMEANS points of the dataset
    for (i = 0; i < KMEANS; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            centroids[i][j] = dataset[i][j];
        }
    }

    // Print the initial values of centroids:
    if (my_rank == 0)
    {
        printf("\nInitial Values of Centroids");
        for (i = 0; i < KMEANS; i++)
        {
            printf("\nC%d\t", i);
            for (j = 0; j < DIM; j++)
            {
                printf("%f\t", centroids[i][j]);
            }
        }
        printf("\n");
    }

    // Assigning a Centroid to each point and Updating Centroids from the points for 10 iterations
    for (i = 0; i < KMEANSITERS; i++)
    {

        // Start measuring time on all ranks
        tstart_total = MPI_Wtime();

        // Initialize (or) reinitialize the values of Sum variables to zero.

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                localSumOfCentroidPoints[j][k] = 0;
                globalSumOfCentroidPoints[j][k] = 0;
            }
            localNumOfPointsForCentroid[j] = 0;
            globalNumOfPointsForCentroid[j] = 0;
        }

        // Assign Centroid closest to each point as assignedCentroid        
        for (j = 0; j < numPointsForMyRank; j++)
        {            
            for (k = 0; k < KMEANS; k++)
            {
                // Start measuring distance calculation time
                tstart_distance = MPI_Wtime();
            
                // Calculate distance of each point to each centroid
                double dimsquares = 0.0;
                for (l = 0; l < DIM; l++)
                {
                    dimsquares = dimsquares + pow((dataset[myStartPoint + j][l] - centroids[k][l]), 2);
                }
                distance = sqrt(dimsquares);

                // End of Distance Calculation Step - Stop measuring time for Distance Calculation
                tend_distance = MPI_Wtime();

                // Calculate time taken for Distance Calculation as (ending time - starting time)
                local_time_taken_distance = local_time_taken_distance + tend_distance - tstart_distance;

                // Check to see if the distance from this centroid is the lowest distance value  
                // for the point. If yes, store index of centroid in assignedCentroid array.

                // Start measuring Updating Centroids time
                tstart_updating_1 = MPI_Wtime();

                if (k == 0)
                {
                    assignedCentroid[j] = k;
                    minimumDistance = distance;
                }
                else
                {
                    if (distance < minimumDistance)
                    {
                        assignedCentroid[j] = k;
                        minimumDistance = distance;
                    }
                }

                // End of Updating Centroids Step 1 - Stop measuring time for updating
                tend_updating_1 = MPI_Wtime();

                // Calculate time taken for Updating Centroids at all ranks
                local_time_taken_updating = local_time_taken_updating + tend_updating_1 - tstart_updating_1;
                        
            }

            // After calculating distance from each Centroid to the point and comparing them, 
            // we have the least value stored in the minimumDistance variable, and the index
            // of this centroid stored in assignedCentroid array. 

            // Use the index stored in assignedCentroid to update the Sum of coordinates and number
            // of points assigned to a particular centroid to help calculate the new Centroid
            // in the next step

            // Start measuring Updating Centroids time
            tstart_updating_2 = MPI_Wtime();

            localNumOfPointsForCentroid[assignedCentroid[j]] = localNumOfPointsForCentroid[assignedCentroid[j]] + 1;
            for (l = 0; l < DIM; l++)
            {
                localSumOfCentroidPoints[assignedCentroid[j]][l] = localSumOfCentroidPoints[assignedCentroid[j]][l] + dataset[myStartPoint + j][l];
            }

            // End of Updating Centroids Step 2 - Stop measuring time for updating
            tend_updating_2 = MPI_Wtime();

            // Calculate time taken for Updating Centroids at all ranks
            local_time_taken_updating = local_time_taken_updating + tend_updating_2 - tstart_updating_2;
        }


        // Calculate new centroid:

        // Use the local sum of coordinates of the points assigned to each centroid stored 
        // in localSumOfCentroidPoints and localNumOfPointsForCentroid to find global sums 
        // and new Centroid

        // Start measuring Updating Centroids time on all ranks
        tstart_updating_3 = MPI_Wtime();

        // Calculate global sums of point coordinates and number of points for each centroid
        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                MPI_Allreduce(&(localSumOfCentroidPoints[j][k]), &(globalSumOfCentroidPoints[j][k]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Allreduce(&(localNumOfPointsForCentroid[j]), &(globalNumOfPointsForCentroid[j]), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        // Use global sums to compute new Centroid
        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                if (globalNumOfPointsForCentroid[j] != 0)
                {
                    centroids[j][k] = globalSumOfCentroidPoints[j][k] / globalNumOfPointsForCentroid[j];
                }
                else
                {
                    // If no points have been assigned to the Centroid, Division by Zero Error may occur
                    // Instead of dividing by 0,re-initialize centroid to Origin.
                    centroids[j][k] = 0;
                }
            }
        }

        // End of Updating Centroids Step - Stop measuring time for updating
        tend_updating_3 = MPI_Wtime();

        // Calculate time taken for Updating Centroids at all ranks
        local_time_taken_updating = local_time_taken_updating + tend_updating_3 - tstart_updating_3;

        // End of both Distance Calculation and Updating Centroids Steps - Stop measuring total time
        tend_total = MPI_Wtime();

        // Calculate time taken to compute distance matrix as (ending time - starting time)
        local_time_taken_total = local_time_taken_total + tend_total - tstart_total;


        // Validation:

        // Print New Centroids:
        if (my_rank == 0)
        {
            printf("\nCentroids in Iteration %d:\n", i);

            for (j = 0; j < KMEANS; j++)
            {
                printf("Centroid %d : ", j);
                for (k = 0; k < DIM; k++)
                {
                    printf("\t%f", centroids[j][k]);
                }
                printf("   : Number of Points assigned to C%d = %d\n", j, globalNumOfPointsForCentroid[j]);
            }
        }

        // Check if Centroids are same on all ranks at each iteration:
        // Perform Maximum and Minimum reduction on the Centroids, if they are both the same and equal
        // to the Centroid, the centroid values are same on all ranks.

        flag = 0;

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                MPI_Reduce(&(centroids[j][k]), &maxCentroid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&(centroids[j][k]), &minCentroid, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

                if (my_rank == 0)
                {
                    if (maxCentroid != minCentroid)
                        flag = 1;
                }
            }
        }

        if (my_rank == 0)
        {
            if (flag == 0)
                printf("Values of Centroids are equal at all ranks!\n");
            else
                printf("Error: Values of Centroids are not equal at all ranks!\n");
        }

    }


    // Use MPI_Reduce to compute maximum time taken among the ranks for Distance Calculation
    MPI_Reduce(&local_time_taken_distance, &global_time_taken_distance, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to compute maximum time taken among the ranks for updating
    MPI_Reduce(&local_time_taken_updating, &global_time_taken_updating, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to compute maximum time taken among the ranks in total
    MPI_Reduce(&local_time_taken_total, &global_time_taken_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print time taken for Distance Calculation, Updating Centroids, and Total Time taken:
    if (my_rank == 0)
    {
        printf("\nTime taken for Distance Calculation = %f\n", global_time_taken_distance);
        printf("Time taken for Updating Centroids = %f\n", global_time_taken_updating);
        printf("Total time taken = %f\n", global_time_taken_total);
    }



    // free dataset
    for (int i = 0; i < N; i++)
    {
        free(dataset[i]);
    }

    free(dataset);
    MPI_Finalize();

    return 0;
}

int importDataset(char *fname, int DIM, int N, double **dataset)
{

    FILE *fp = fopen(fname, "r");

    if (!fp)
    {
        printf("Unable to open file\n");
        return (1);
    }

    char buf[4096];
    int rowCnt = 0;
    int colCnt = 0;
    while (fgets(buf, 4096, fp) && rowCnt < N)
    {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[rowCnt][colCnt] = tmp;

        while (field)
        {
            colCnt++;
            field = strtok(NULL, ",");

            if (field != NULL)
            {
                double tmp;
                sscanf(field, "%lf", &tmp);
                dataset[rowCnt][colCnt] = tmp;
            }
        }
        rowCnt++;
    }

    fclose(fp);
    return 0;
}

