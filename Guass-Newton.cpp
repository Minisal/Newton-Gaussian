#include <iostream>
#include <eigen3/Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <chrono>
/*

# Derive
For a nonlinear least squares problem
	x = argmin( 1/2 * ||f(x)||^2 )                          ------ (1)

Use Taylor expansion to obtain the first-order linear approximation of f(x)
	f(x+delta_x) = f(x)+f'(x)*delta_x = f(x)+J(x)*delta_x   ------ (2)

Bring (2) into (1) 
	1/2 * ||f(x+delta_x)||^2                                ------ (3)
  = 1/2 * { f(x)^T*f(x) + 2*f(x)^T*J(x)*delta_x + delta_x^T*J(x)^T*J(x)*delta_x}

Take the derivative of (3) and set the derivative to zero
	J(x)^T*J(x)*delta_x = -J(x)^T*f(x)                      ------ (4)

Let H = J^T*J, B = -J^T*j, formula (4) is
	H * delta_x = B                                         ------ (5)

Solving equation (5) to obtain the adjustment increment delta_x


# Steps

1. Given initial value x_0
2. For the kth iteration, calculate Jacobian J, matrix H, B;
   calculate the increment according to formula (5)
3. If the delta_x_k is small enough, stop the iteration;
   otherwise, update the x_k+1 = x_k + delta_x_k
4. Execute 2.&3. cyclically until the maximum number of cycles is reached,
   or the termination condition of 3. is satisfied.



# Problem
nonlinear function : y = exp(ax^2+bx+c)
for n sets of observation data {x, y} , find the coefficient X=[a,b,c]^T

# Analysis
Let f(X) = y - exp(ax^2 + bx + c)
N sets of data can form a large nonlinear equation matrix 
	F(X) =	[ y_1 - exp(ax_1^2 + bx_1 + c) ]
	    	[ y_N - exp(ax_N^2 + bx_N + c) ]

construct a least squares problem
	x = argmin 1/2*||F(X)||^2

Find the Jacobian to solve this problem 
	J(x) = [ -x_1^2 exp(ax_1^2+bx_1+c)  -x_1^2 exp(ax_1^2+bx_1+c)  -x_1^2 exp(ax_1^2+bx_1+c) ]
		   |            ...                         ...                       ...            |
		   [ -x_N^2 exp(ax_N^2+bx_N+c)  -x_N^2 exp(ax_N^2+bx_N+c)  -x_N^2 exp(ax_N^2+bx_N+c) ]

*/



/* timer */
class Runtimer{
public:
	inline void start()
	{
		t_s_ = std::chrono::steady_clock::now();
	}

	inline void stop()
	{
		t_e_ = std::chrono::steady_clock::now();
	}

	inline double duration()
	{
		return std::chrono::duration_cast<std::chrono::duration<double>>(t_e_-t_s_).count()*1000.0;
	}

private:
	std::chrono::steady_clock::time_point t_s_; // start time point
	std::chrono::steady_clock::time_point t_e_; // end / stop time point
};


/* optimizer */
class CostFunction{
public:
	CostFunction(double* a, double* b, double* c, int max_iter, double min_step, bool is_out):
	a_(a), b_(b), c_(c), max_iter_(max_iter), min_step_(min_step), is_out_(is_out) {}

	void addObservation(double x, double y)
	{
		std::vector<double> ob;
		ob.push_back(x);
		ob.push_back(y);
		obs_.push_back(ob);
	}

	void calcJ_fx()
	{
		J_ .resize(obs_.size(), 3);
		fx_.resize(obs_.size(), 1);

		for (size_t i=0; i<obs_.size(); ++i)
		{
			std::vector<double>& ob = obs_.at(i);
			double& x = ob.at(0);
			double& y = ob.at(1);
			double j1 = -x*x*exp(*a_ * x*x + *b_ * x + *c_);
			double j2 =   -x*exp(*a_ * x*x + *b_ * x + *c_);
			double j3 =     -exp(*a_ * x*x + *b_ * x + *c_);
			J_(i, 0) = j1;
			J_(i, 1) = j2;
			J_(i, 2) = j3;
			fx_(i, 0) = y - exp(*a_ * x*x + *b_ * x + *c_);
		}
	}

	void calcH_b()
	{
		H_ =  J_.transpose() * J_;
		B_ = -J_.transpose() * fx_;
	}

	void calcDeltax()
	{
		deltax_ = H_.ldlt().solve(B_);
	}

	void updateX()
	{
		*a_ += deltax_(0);
		*b_ += deltax_(1);
		*c_ += deltax_(2);
	}

	double getCost()
	{
		Eigen::MatrixXd cost = fx_.transpose() * fx_;
		return cost(0,0);
	}

	void solveByGaussNewton()
	{
		double sumt = 0;
		bool is_conv = false;
		for (size_t i=0; i<max_iter_; i++)
		{
			Runtimer t;
			t.start();
			calcJ_fx();
			calcH_b();
			calcDeltax();
			double delta = deltax_.transpose() * deltax_;
			t.stop();
			if (is_out_)
			{
				std::cout << "Iter: "   << std::left << std::setw(3)  << i 
						  << " Result: "<< std::left << std::setw(10) << *a_ << " " 
						  				<< std::left << std::setw(10) << *b_ << " " 
						  				<< std::left << std::setw(10) << *c_ 
						  << " step: "  << std::left << std::setw(14) << delta 
						  << " cost: "  << std::left << std::setw(14) << getCost() 
						  << " time: "  << std::left << std::setw(14) << t.duration()  
						  << " total_time: "<< std::left <<std::setw(14) << (sumt += t.duration()) << std::endl;		  				
			}

			if (delta < min_step_)
			{
				is_conv = true;
				break;
			}
			updateX();
		}

		if (is_conv == true) std::cout << "\nConverged\n";
		else 				 std::cout << "\nDiverged\n\n";
	}

	Eigen::MatrixXd fx_;
	Eigen::MatrixXd J_; 
	Eigen::MatrixXd H_;
	Eigen::Vector3d B_;
	Eigen::Vector3d deltax_;
	std::vector< std::vector<double> >obs_; // observation points
	double *a_, *b_, *c_;

	int max_iter_;
	double min_step_;
	bool is_out_;
};


int main(int argc, char** argv)
{
	const double aa = 0.1, bb = 0.5, cc = 2;
	double a=0.0, b=0.0, c=0.0; // init value

	/* Construction problem */
	CostFunction cost_func(&a, &b, &c, 50, 1e-10, true);

	/* Manufacturing data */
	const size_t N = 100;
	cv::RNG rng(cv::getTickCount());
	for(size_t i=0; i<N; i++)
	{
		/* Data with Gaussion noise */
		double x = rng.uniform(0.0, 1.0);
		double y = exp(aa*x*x + bb*x + cc) + rng.gaussian(0.05);

		/* Add to observation */
		cost_func.addObservation(x, y);
	}

	cost_func.solveByGaussNewton();
	return 0;
}