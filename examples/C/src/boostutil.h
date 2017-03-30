// utility functions for boost matrices/vectors

#include <fstream>

namespace ublas = boost::numeric::ublas;
using namespace std;

double vsum(ublas::vector<double> v)
{
  double s = 0.0;
  for(unsigned int i = 0; i < v.size(); i++)
    s += v(i);
  return s;
}

double vmax(ublas::vector<double> v)
{
  return *max_element(v.begin(), v.end());
}

double vmin(ublas::vector<double> v)
{
  return *min_element(v.begin(), v.end());
}

void zero(ublas::vector<double> & v)
{
  for(unsigned int i = 0; i < v.size(); i++)
    v(i) = 0.0;
}

void loadtxt(string fname, ublas::matrix<double> & M)
{
  ifstream f (fname.c_str());
    
  if(!f.is_open())
  {
    cerr << "Error opening file " << fname << ". Exiting..." << endl;
    exit(EXIT_FAILURE);
  }

  for (unsigned int i = 0; i < M.size1(); i++)
    for (unsigned int j = 0; j < M.size2(); j++)
      f >> M(i,j);

  f.close(); 
}

void savetxt(string fname, ublas::matrix<double> & M)
{
  ofstream f (fname.c_str());

  if(!f.is_open())
  {
    cerr << "Error opening file " << fname << ". Exiting..." << endl;
    exit(EXIT_FAILURE);
  }

  for (unsigned int i = 0; i < M.size1(); i++)
  {
    for (unsigned int j = 0; j < M.size2(); j++)
    {
      f << M(i,j);
      if(j < M.size2()-1) f << " ";
      else f << endl;
    }
  }

  f.close(); 
}
