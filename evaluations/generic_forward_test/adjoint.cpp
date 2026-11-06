// Independent extension; native_wave.cpp is compiled unchanged alongside this.
#include <algorithm>
#include <vector>
extern "C" int wave_forward(int,int,double,double,const double*,const double*,const double*,int,const int*,int,double*,double*);
extern "C" int wave_extended_forward(int nx,int nt,double dt,double dx,const double* c2,const double* damping,const double* src,int si,const int* ri,int nr,double* data,double* history) {
    return wave_forward(nx,nt,dt,dx,c2,damping,src,si,ri,nr,data,history);
}
// Generic observation pair, including endpoints and duplicate receivers.
extern "C" void wave_sample(int nx,int nt,const int* ri,int nr,const double* hist,double* data) {
    for(int t=0;t<nt;++t) for(int r=0;r<nr;++r) data[t*nr+r]=hist[(t+1)*nx+ri[r]];
}
extern "C" void wave_sample_transpose(int nx,int nt,const int* ri,int nr,const double* bar,double* impulses) {
    std::fill(impulses,impulses+(nt+1)*nx,0.);
    for(int t=0;t<nt;++t) for(int r=0;r<nr;++r) impulses[(t+1)*nx+ri[r]]+=bar[t*nr+r];
}
extern "C" void wave_adjoint(int nx,int nt,double dt,double dx,const double* c2,const double* damping,const int* ri,int nr,const double* history,const double* bar,double* gc2) {
    const double a=dt*dt/(dx*dx);
    std::fill(gc2,gc2+nx,0.);
    std::vector<double> next(nx,0.),current(nx,0.),previous(nx,0.);
    for(int t=nt-1;t>=0;--t) {
        for(int r=0;r<nr;++r) next[ri[r]]+=bar[t*nr+r];
        const double* u=history+t*nx;
        for(int i=1;i<nx-1;++i) {
            const double x=damping[i]*next[i];
            gc2[i]+=a*(u[i-1]-2.*u[i]+u[i+1])*x;
            current[i]+=2.*x;
            const double s=a*c2[i]*x;
            current[i]-=2.*s;
            if(i>1) current[i-1]+=s;
            if(i<nx-2) current[i+1]+=s;
            previous[i]=-x;
        }
        next.swap(current);
        current.swap(previous);
        std::fill(previous.begin(),previous.end(),0.);
    }
}
// Analytic tangent is a check-only JVP, not used by the production gradient.
extern "C" void wave_tangent(int nx,int nt,double dt,double dx,const double* c2,const double* damping,const int* ri,int nr,const double* history,const double* dc2,double* tangent) {
    const double a=dt*dt/(dx*dx);
    std::vector<double> previous(nx,0.),current(nx,0.),next(nx,0.);
    for(int t=0;t<nt;++t) {
        std::fill(next.begin(),next.end(),0.);
        const double* u=history+t*nx;
        for(int i=1;i<nx-1;++i) {
            const double lap=current[i-1]-2.*current[i]+current[i+1];
            const double ulap=u[i-1]-2.*u[i]+u[i+1];
            next[i]=damping[i]*(2.*current[i]-previous[i]+a*(c2[i]*lap+dc2[i]*ulap));
        }
        for(int r=0;r<nr;++r) tangent[t*nr+r]=next[ri[r]];
        previous.swap(current);current.swap(next);
    }
}
