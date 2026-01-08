Gross-Pitaevskii Equation Solver (3-Dimensional Spherically Symmetric Case)

gpe

    common.py

        evaluate_dt_list for transforming the string structure, which is "float"*dr*dr to float. This did not affect directly using float.

        initial_phi for initial function which is r * np.exp(-r*r).

        normalize for normalization.

        expectation_value_energy for calculating the expectation value of energy.

        full_grid for making a domain. It is extended from -R to R. This is useful for fft.

        V_full for potential + non-linear part of GPE. There is also adjusment for avoiding singularities.

    solvers.py

        Here, functions represent one step of the process. There are three methods: Time Splitting Spectral Method, Forward-Euler Method, and Crank-Nicolson Method 

        Time Splitting Spectral Method

            Based on Strang Splitting. The operations run with fft. It is very stable. (TSSM_kinetic_step, TSSM_step)

        Forward-Euler Method

            Based on a series expansion. Just the first order of imaginary time is considered. It is not very stable, but the fastest one. (laplacian, FE_step) 

        Crank-Nicolson Method

            It is also based on a series expansion, but in a different way. It is very stable. (CN_step)

    runners.py

        Run every method up to some thresholds. These thresholds are the maximum iteration number, relative energy differences, and for a given energy limit.

scripts

    input.json

        methods are {"FE","CN","TSSM"}, where "FE" refers to Forward-Euler Method, "CN" refers to Crank-Nicolson Method, "TSSM" refers to Time Splitting Spectral Method.

        g_list stands for coupling constants.

        R_list stands for domain size.

        dt_list stands for step-size.

        numeric part is for the thresholds and checks:

            N stands for how many division have spectral part.

            tol stands for at which relative energy difference code stops.

            max_iter is for limiting maximum imaginary time steps.

            renorm_every is for normalizing wavefunction at each given step by this number. 

            report_every is for looking at the relative energy differences at each given step by this number. 

            E_aim is for stopping code for this energy limit.

        out_dir is where data collected.

    run_sweeps.py

        This py is run methods by runners.py for different parameters, and organize them for reading easily. Lastly saving data in the format of npz.

    plots.py

        This py plots data with dfferent comparisons:

            Different methods.

            Different coupling constants.

            Different spectral domains.
        
        Also, there is a code for csv file to the comparison of energy expectation values.
