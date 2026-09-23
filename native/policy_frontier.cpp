// Experimental implementations of random SA, granular search and ALNS-style
// destroy/repair. Original implementation, not a reproduction of a paper's code.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Value { double t=0,e=0,d=0,late=0,score=0; };
struct Problem {
    int n; const double *travel,*service,*demand,*early,*late,*coef; double capacity;
    Value eval(const std::vector<int>& r) const {
        Value v; double load=0,now=0;
        for (int j:r) load+=demand[j];
        for (size_t k=1;k<r.size();++k) {
            int i=r[k-1],j=r[k];double t=travel[i*n+j];
            v.t+=t;v.e+=t*(1+(capacity>0?load/capacity:0));
            now=std::max(now+t,early[j]);v.late+=std::max(0.,now-late[j]);
            now+=service[j]; if(j) v.d+=now; load-=demand[j];
        }
        v.score=coef[0]*v.t+coef[1]*v.d+coef[2]*v.e;return v;
    }
};
extern "C" int policy_frontier(int n,const double* travel,const double* service,
    const double* demand,const double* early,const double* late,double capacity,
    const int* initial,const double* coef,double tlimit,double elimit,int method,
    uint64_t seed,int count,const double* checkpoints,int* routes,double* stats) {
    Problem p{n,travel,service,demand,early,late,coef,capacity};
    std::vector<int> current(initial,initial+n+1),best=current,candidate;
    Value cv=p.eval(current),bv=cv;std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.,1.);
    std::vector<std::vector<int>> neighbours(n);
    for(int i=1;i<n;++i) {
        for(int j=1;j<n;++j)if(i!=j)neighbours[i].push_back(j);
        std::sort(neighbours[i].begin(),neighbours[i].end(),[&](int a,int b){
            return std::min(travel[i*n+a],travel[a*n+i])<std::min(travel[i*n+b],travel[b*n+i]);});
        if(neighbours[i].size()>32)neighbours[i].resize(32);
    }
    std::array<double,3> weights{1,1,1};
    uint64_t attempts=0,valid=0,accepted=0,evaluations=1;
    auto start=Clock::now();int recorded=0;
    while(recorded<count) {
        double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
        if(elapsed>=checkpoints[recorded] || n<3) {
            std::copy(best.begin(),best.end(),routes+recorded*(n+1));
            double* s=stats+recorded*7;
            s[0]=elapsed;s[1]=attempts;s[2]=valid;s[3]=accepted;s[4]=evaluations;s[5]=bv.score;s[6]=bv.late;
            ++recorded;continue;
        }
        if(attempts && attempts%1000==0){current=best;cv=bv;}
        candidate=current;int op=-1;
        if(method==2 && unit(rng)<.2 && n>4) {
            double draw=unit(rng)*std::accumulate(weights.begin(),weights.end(),0.);
            op=0;while(op<2 && (draw-=weights[op])>0)++op;
            int k=2+rng()%std::min(5,n-2),pivot=1+rng()%(n-1);
            std::vector<int> removed;
            for(int z=0;z<k && candidate.size()>3;++z) {
                int pos=1+rng()%(candidate.size()-2);
                if(op==1) {
                    double nearest=1e300;
                    for(int j=1;j<(int)candidate.size()-1;++j){
                        double c=travel[current[pivot]*n+candidate[j]];
                        if(c<nearest){nearest=c;pos=j;}
                    }
                } else if(op==2) {
                    double worst=-1e300;
                    for(int j=1;j<(int)candidate.size()-1;++j){
                        int a=candidate[j-1],b=candidate[j],c=candidate[j+1];
                        double cost=travel[a*n+b]+travel[b*n+c]-travel[a*n+c];
                        if(cost>worst){worst=cost;pos=j;}
                    }
                }
                removed.push_back(candidate[pos]);candidate.erase(candidate.begin()+pos);
            }
            std::shuffle(removed.begin(),removed.end(),rng);
            for(int node:removed){
                int bestpos=1;double insertion=1e300;
                for(int j=1;j<(int)candidate.size();++j){
                    auto trial=candidate;trial.insert(trial.begin()+j,node);
                    auto v=p.eval(trial);++evaluations;
                    double key=v.score+100*v.late/std::max(1.,tlimit);
                    if(key<insertion){insertion=key;bestpos=j;}
                }
                candidate.insert(candidate.begin()+bestpos,node);
            }
        } else {
            int a=1+rng()%(n-1),b=1+rng()%(n-1),kind=rng()%3;
            if(method && unit(rng)<.9 && !neighbours[current[a]].empty()){
                int node=neighbours[current[a]][rng()%neighbours[current[a]].size()];
                b=std::find(current.begin(),current.end(),node)-current.begin();
                if(kind==0 && unit(rng)<.5)b=std::min(n-1,b+1);
            }
            if(a==b)continue;
            if(kind==0){int node=candidate[a];candidate.erase(candidate.begin()+a);candidate.insert(candidate.begin()+b,node);}
            else if(kind==1)std::swap(candidate[a],candidate[b]);
            else {if(a>b)std::swap(a,b);std::reverse(candidate.begin()+a,candidate.begin()+b+1);}
        }
        ++attempts;auto v=p.eval(candidate);++evaluations;double reward=.25;
        if(v.late<=1e-9 && v.t<=tlimit+1e-9 && v.e<=elimit+1e-9){
            ++valid;
            double temperature=std::max(1e-6,.003*(1.-(attempts%50000)/50000.));
            bool take=v.score<cv.score || unit(rng)<std::exp((cv.score-v.score)/temperature);
            if(take){current=candidate;cv=v;++accepted;reward=1.;}
            if(v.score<bv.score){best=candidate;bv=v;reward=5.;}
        }
        if(op>=0)weights[op]=.8*weights[op]+.2*reward;
    }
    return 0;
}
