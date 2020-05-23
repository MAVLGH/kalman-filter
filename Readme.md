# Welcome to the Kalman Filter repository
This is an implementation of a linear kalman filter considering a discrete linear dynamical system.


## Installation
* Install the requirements:


    pip install -r requirements.txt
    
    
## Usage

* The main.py script contains an example that uses a dynamical system and a kalman filter estimator
* From DynamicalSystem class is assumed known a, b, c, r_ww, r_vv.
* From Kalman Filter estimator is assumed known a, b, c, r_ww, r_vv. x_t_t_ini and p_t_t_ini are initial mean and covariance estimations of the system.

## References

* James V. Candy - Bayesian Signal Processing, Classical, Modern, and Particle Filtering Methods. p.155.

## Author

Mario Vergara - mario.a.vergara.l@gmail.com 