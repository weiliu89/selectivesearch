#include <cstdio>
#include <algorithm>
#include <vector>
#include <deque>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "mex.h"
using namespace std;

// Usage: [beta, bad] = LLC_pooling_mex(x, y, llc_codes, boxes, pyramid);
//   Assume the the box coordinates are [top, left, bottom, right].


class Maxer
{
	public:
		int size;
		double *val;

		Maxer(int s){size = s; val = (double*) calloc(s, sizeof(double));}
		~Maxer(){free(val);}
		void set_values (int n, mwIndex *idx, double *new_val);
		void update_values (int n, mwIndex *idx, double *new_val);
		void merge_values(Maxer &a);
		void dump_values(double *dest, int start, int n);
		double dump_values_vec(deque<double> &betaDeque, deque<int> &irDeque, int startIdx, bool pushBack);
		void disp();
};
void Maxer::dump_values(double *dest, int start, int n)
{
	double *p = val+start;
	memcpy(dest, p, n*sizeof(double));
}

double Maxer::dump_values_vec(deque<double> &betaDeque, deque<int> &irDeque, int startIdx, bool pushBack)
{
	double sumVal = 0.0;
	if (pushBack)
	{
		for (int i=0; i<size; i++)
		{
			double ival = val[i];
			if (ival != 0.0)
			{
				betaDeque.push_back(ival);
				irDeque.push_back(startIdx+i);
				sumVal += ival*ival;
			}
		}
	}
	else
	{
		for (int i=size-1; i>=0; i--)
		{
			double ival = val[i];
			if (ival != 0.0)
			{
				betaDeque.push_front(ival);
				irDeque.push_front(startIdx+i);
				sumVal += ival*ival;
			}
		}
	}
	return sumVal;
}

void Maxer::set_values(int n, mwIndex *idx, double *new_val)
{
	memset(val, 0, size * sizeof(double));
	for(; n > 0; n--)
		val[*idx++] = *new_val++;
}

void Maxer::update_values(int n, mwIndex *idx, double *new_val)
{
	for(; n > 0; n--, idx++)
		val[*idx] = max(val[*idx], *new_val++);
}

void Maxer::merge_values(Maxer &a)
{
	assert(size == a.size);
	int n = size;
	double *new_val = a.val;
	for(double *p = val; n > 0; n--, p++, new_val++)
		*p = max(*p, *new_val);
}

void Maxer::disp()
{
	for(int i = 0; i < size; i++)
		if(val[i] != 0.0)
			fprintf(stderr, "%d:%f\n", i+1, val[i]);
}

double *x = NULL;
double *y = NULL;
int npoints = 0;
const mxArray *codes;
vector<int> scale_start;   // starting index to x and to y in each scale
vector<int> nrows, ncols;  // number of rows and columns in each scale
vector<double> step_x, step_y; // the interval between closest feature points
vector<double> min_x, max_x, min_y, max_y; 

int Pool(Maxer *mxr, double left, double top, double right, double bot)
{
	// Get pointer for the sparse codes
	mwIndex *ir = mxGetIr(codes);
	mwIndex *jc = mxGetJc(codes);
	double  *data = mxGetPr(codes);

	// Find the indices of points that falls into the box
	vector<int> idx;

	// pool in each scale
	for(int s = 0; s < scale_start.size(); s++)
	{
		// boundary check
		if(left > max_x[s] || right < min_x[s] || top > max_y[s] || bot < min_x[s])
			continue;

		// find the left most top most point
		int col_start = max(0, (int)ceil( ( left  - min_x[s] ) / step_x[s] ));
		int col_stop  = min(ncols[s], (int)ceil( ( right - min_x[s] + 0.0001) / step_x[s]));
		int row_start = max(0, (int)ceil( ( top   - min_y[s] ) / step_y[s] ));
		int row_stop  = min(nrows[s], (int)ceil( ( bot   - min_y[s] + 0.0001) / step_y[s]));

		// add indices to the vector
		for(int c = col_start; c < col_stop; c++)
			for(int r = row_start; r < row_stop; r++)
				idx.push_back( scale_start[s] + c * nrows[s] + r);
	}

	if(idx.size() == 0)
		return 1;

	int p = idx[0];
	long head = jc[p];
	mxr->set_values(jc[p+1] - jc[p], ir+head, data+head);
	for(int i = 1; i < idx.size(); i++)
	{
		p = idx[i];
		head = jc[p];
		mxr->update_values(jc[p+1] - jc[p], ir+head, data+head);
	}
	return 0;
}

void Normalize(double *head, int n)
{
	double s = 0;
	double *p = head;
	for(int i = 0; i < n; i++, p++)
		if( *p != 0.0 )
			s += (*p) * (*p);
	s = sqrt(s);
	for(; n > 0; n--)
		*head++ /= s;
}

void NormalizeDeque(deque<double> &betaDeque, double sumVals)
{
	for (int i=0; i<betaDeque.size(); i++)
		betaDeque[i] /= sumVals;
}

void mexFunction(int nlhs, mxArray *plhs[], 
		int nrhs, const mxArray *prhs[])
{
	x = mxGetPr(prhs[0]); // x-coordinates of the feature points
	y = mxGetPr(prhs[1]); // y-coordinates of the feature points
	double *scales = mxGetPr(prhs[2]); // scale of the feature points
	codes = prhs[3];    
	double *pbox = mxGetPr(prhs[4]); // pointer to boxes
	double *pyramid = mxGetPr(prhs[5]); // the pyramid size

	int nlevel = mxGetN(prhs[5]);
	int nbins  = 0;
	for(int i = 0; i < nlevel; i++)
		nbins += pyramid[i] * pyramid[i];
	int dim = mxGetM(codes) * nbins;
	int code_dim = mxGetM(codes);

	// setup pointers to box coordinates
	int nbox = mxGetM(prhs[4]);
	double *top = pbox, *left = pbox+nbox, *bot = pbox+2*nbox, *right = pbox+3*nbox;

	// feature point info
	npoints = mxGetN(prhs[0]);

	// find scale starts
	double *q = scales+1;
	double pre = scales[0];
	scale_start.push_back(0);
	for(int i=1; i<npoints; i++, q++)
	{
		if(*q != pre)
		{
			scale_start.push_back(i);
			pre = *q;
		}
	}

	// find nrows and ncols, step and limits along x and y
	for(int i=0; i<scale_start.size(); i++)
	{
		int start = scale_start[i];
		pre = x[start];
		int cnt = 0;
		while(pre == x[start+(++cnt)])
			;
		nrows.push_back(cnt);
		if(i+1<scale_start.size())
		{
			ncols.push_back((scale_start[i+1]-scale_start[i])/cnt);
			max_x.push_back(x[scale_start[i+1]-1]);
		}
		else
		{
			ncols.push_back((npoints - scale_start[i])/cnt);
			max_x.push_back(x[npoints-1]);
		}
		step_y.push_back(y[start+1]-y[start]);
		step_x.push_back(x[start+nrows.back()] - x[start]);
		min_x.push_back(x[start]);
		min_y.push_back(y[start]);
		max_y.push_back(y[start+nrows.back()-1]);
	}

	// assume 2 levels
	Maxer mxr_L1(code_dim);
	Maxer mxr_L2(code_dim);
	vector<deque<double> > betaVecs;
	vector<deque<int> > irVecs;
	vector<int> jcVec;
	vector<int> goodIdxVec;
	bool isBad;
	int nzmax = 0;
	for(int i = 0; i < nbox; i++)
	{
		isBad = false;
		int m = pyramid[1];
		double side_x = (double)(right[i] - left[i] + 1) / m;
		double side_y = (double)(bot[i] - top[i] + 1) / m;
		deque<double> betaDeque;
		betaDeque.clear();
		deque<int> irDeque;
		irDeque.clear();
		double sumVals = 0.0;
		for(int j = 0; j < m; j++) // row
		{
			for(int k = 0; k < m; k++) // col
			{
				// getting the coordinate of each window
				double s_left = left[i] + side_x * k;
				double s_right= s_left + side_x - 0.001;
				double s_top  = top[i] + side_y * j;
				double s_bot  = s_top + side_y - 0.001;

				if(Pool(&mxr_L2, s_left, s_top, s_right, s_bot) != 0)
				{
					isBad = true;
					break;
				}
				else
				{
					if(j == 0 && k == 0)
						memcpy(mxr_L1.val, mxr_L2.val, sizeof(double)*code_dim);
					else
						mxr_L1.merge_values(mxr_L2);
					sumVals += mxr_L2.dump_values_vec(betaDeque, irDeque, (j*m+k+1)*code_dim, true);
				}
			}
			if(isBad)
				break;
		}
		if(!isBad)
		{
			sumVals += mxr_L1.dump_values_vec(betaDeque, irDeque, 0, false);
			sumVals = sqrt(sumVals);
			NormalizeDeque(betaDeque, sumVals);
			jcVec.push_back(nzmax);
			nzmax += betaDeque.size();
			betaVecs.push_back(betaDeque);
			irVecs.push_back(irDeque);
			goodIdxVec.push_back(i+1);
		}
	}
	jcVec.push_back(nzmax);
	int nGoods = goodIdxVec.size();

	// setup result matrices
	plhs[0] = mxCreateSparse(dim, nGoods, nzmax, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(nGoods, 1, mxREAL);

	// output results
	double* beta = mxGetPr(plhs[0]);
	mwIndex* irs = mxGetIr(plhs[0]);
	mwIndex* jcs = mxGetJc(plhs[0]);
	int count = 0;
	for (int i=0; i<betaVecs.size(); i++)
	{
		for (int j=0; j<betaVecs[i].size(); j++)
		{
			beta[count] = betaVecs[i][j];
			irs[count] = irVecs[i][j];
			count++;
		}
	}

	for (int i=0; i<jcVec.size(); i++)
		jcs[i] = jcVec[i];

	double* goodIdx = mxGetPr(plhs[1]);
	for (int i=0; i<nGoods; i++)
		goodIdx[i] = goodIdxVec[i];

	// clear global variables
	scale_start.clear();
	nrows.clear();
	ncols.clear();
	step_x.clear();
	step_y.clear();
	min_x.clear();
	min_y.clear();
	max_x.clear();
	max_y.clear();

	betaVecs.clear();
	irVecs.clear();
	jcVec.clear();
	goodIdxVec.clear();
}
