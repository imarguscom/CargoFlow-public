// Fixed-endpoint directed Hamiltonian-path DP (Held-Karp recurrence).
// Each proposed block must improve travel and pass the complete original
// integer service-start schedule. The DP relaxation alone is never accepted.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" int cargoflow_window_dp(int n, int *route, const int64_t *cost,
 const int64_t *service, const int64_t *early, const int64_t *late,
 int width, double seconds, int offset, int *scans) {
 if(n<3 || width<2 || width>12 || seconds<=0) return 0;
 auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
 int accepted=0; *scans=0;
 const int k=std::min(width,n-1), positions=n-k;
 const int states=1<<k;
 const int64_t INF=std::numeric_limits<int64_t>::max()/8;
 std::vector<int64_t> dp(states*k);
 std::vector<int> prev(states*k), proposal(n+1), nodes(k);
 auto edge=[&](int a,int b){return cost[a*n+b];};
 auto valid=[&](const std::vector<int>& path){
   int64_t clock=0;
   if(early[0]>0 || late[0]<0)return false;
   for(int i=1;i<=n;i++){
     int a=path[i-1],b=path[i];
     clock=std::max(clock+edge(a,b),early[b]);
     if(clock>late[b])return false;
     clock+=service[b];
   }
   return true;
 };
 bool changed=true;
 while(changed && std::chrono::steady_clock::now()<deadline){
   changed=false;
   for(int step=0;step<positions;step++){
     if(std::chrono::steady_clock::now()>=deadline)return accepted;
     int start=(step+offset)%positions;
     for(int j=0;j<k;j++)nodes[j]=route[start+1+j];
     int left=route[start],right=route[start+k+1];
     std::fill(dp.begin(),dp.end(),INF);
     std::fill(prev.begin(),prev.end(),-1);
     for(int j=0;j<k;j++)dp[(1<<j)*k+j]=edge(left,nodes[j]);
     for(int mask=1;mask<states;mask++){
       if((mask&255)==0 && std::chrono::steady_clock::now()>=deadline)return accepted;
       for(int j=0;j<k;j++)if(mask&(1<<j)){
         int64_t base=dp[mask*k+j];
         if(base==INF)continue;
         for(int l=0;l<k;l++)if(!(mask&(1<<l))){
           int at=(mask|(1<<l))*k+l;
           int64_t value=base+edge(nodes[j],nodes[l]);
           if(value<dp[at]){dp[at]=value;prev[at]=j;}
         }
       }
     }
     (*scans)++;
     int64_t best=INF,original=0;int end=-1;
     for(int j=0;j<k;j++){
       int64_t value=dp[(states-1)*k+j]+edge(nodes[j],right);
       if(value<best){best=value;end=j;}
     }
     for(int i=start;i<=start+k;i++)original+=edge(route[i],route[i+1]);
     if(best>=original)continue;
     std::copy(route,route+n+1,proposal.begin());
     int mask=states-1;
     for(int pos=k;pos>=1;pos--){
       proposal[start+pos]=nodes[end];
       int parent=prev[mask*k+end];mask^=1<<end;end=parent;
     }
     if(valid(proposal)){
       std::copy(proposal.begin(),proposal.end(),route);
       accepted++;changed=true;
     }
   }
   offset=(offset+1)%positions;
 }
 return accepted;
}
