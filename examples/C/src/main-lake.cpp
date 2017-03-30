/* LakeProblem_5Obj_1Const_Stoch.cpp
   
  Riddhi Singh, May, 2014
  The Pennsylvania State University
  rus197@psu.edu

  Adapted by Tori Ward, July 2014 
  Cornell University
  vlw27@cornell.edu

  Adapted by Jonathan Herman and David Hadka, Sept-Dec 2014, March 2017
  Cornell University and The Pennsylvania State University

  A multi-objective represention of the lake model from Carpenter et al., 1999
  This simulation is designed for optimization with Rhodium.

  Stochasticity is introduced by natural phosphorous inflows. 
  These follow a lognormal distribution with specified mean and stdev.

  Decision Vector 
    pollution_limit : anthropogenic pollution flow at previous time step - Size 100, Bounds (0.0,0.1)

  Objectives
  1. minimize the maximum Phosphorous averaged over all lognormal draws in a given time period
  2. maximize expected benefit from pollution
  3. maximize the probability of meeting an inertia constraint
  4. maximize Reliability

*/

#include <stdio.h>
#include <unistd.h>
#include <sstream>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/tools/roots.hpp>
#include "boostutil.h"
#include "generate-scenarios.h"

#define nDays 100
#define nSamples 100
#define alpha 0.4 // utility from pollution
#define beta 0.08 // eutrophic cost
#define reliability_threshold 0.85
#define inertia_threshold (-0.02)

namespace ublas = boost::numeric::ublas;
namespace tools = boost::math::tools;
using namespace std;

extern "C" {

// utility variables and functions for computing pcrit
double root_b;
double root_q;

double root_function(double x) {
  return pow(x,root_q)/(1+pow(x,root_q)) - root_b*x;
}

bool root_termination(double min, double max) {
  return abs(max - min) <= 0.000001;
}

double compute_pcrit(double b, double q) {
  root_b = b;
  root_q = q;
  std::pair<double, double> result = tools::bisect(root_function, 0.01, 1.0, root_termination);
  return (result.first + result.second) / 2;
}

// define the lake problem
void lake_problem(double* pollution_limit, double b, double q, double mean, double stdev, double delta,
    double* max_P, double* utility, double* inertia, double* reliability) 
{
  // initialize the arrays storing computed values
  ublas::vector<double> average_daily_P(nDays);
  ublas::vector<double> discounted_benefit(nSamples);
  ublas::vector<double> days_inertia_met(nSamples);
  ublas::vector<double> days_pcrit_met(nSamples);

  zero(average_daily_P);
  zero(discounted_benefit);
  zero(days_inertia_met);
  zero(days_pcrit_met);
  
  // determine the value of pcrit
  double pcrit = compute_pcrit(b, q);

  for (int s = 0; s < nSamples; s++)
  {   
    // randomly generated natural phosphorous inflows
    ublas::vector<double> P_inflow = generateSOW(nDays, mean, stdev);

    double X = 0.0; // lake state 

    //implement the lake model from Carpenter et al. 1999
    for (int i = 0; i < nDays; i++)
    {
      // new state: previous state - decay + recycling + pollution
      X = X*(1-b) + pow(X,q)/(1+pow(X,q)) + pollution_limit[i] + P_inflow(i);

      average_daily_P(i) += X/nSamples;

      discounted_benefit(s) += alpha*pollution_limit[i]*pow(delta,i);

      if(i > 0 && pollution_limit[i]-pollution_limit[i-1] > inertia_threshold)
        days_inertia_met(s) += 1;

      if(X < pcrit)
        days_pcrit_met(s) += 1;
    }
  }

  // calculate objectives
  *max_P = vmax(average_daily_P);
  *utility = vsum(discounted_benefit)/nSamples;
  *inertia = vsum(days_inertia_met)/((nDays-1)*nSamples);
  *reliability = vsum(days_pcrit_met)/(nDays*nSamples);
}

} // extern "C"
