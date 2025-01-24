#include <iostream>
#include<iomanip>
using namespace std;
const double EPS=1E-6;
int main()
{
     int n,item;
     double e;
     int no;
     cin >> no;
     e=1;
     n=1;
     item=1;
     if (105<no&&no<217){
         do
         {
             item=1;
             for(int i=1; i<=n; i++)
                 item*=i;
             e+=1.0/item;
             n++;
         }
         while((1.0/item)>=EPS);
     cout<<"e="<<fixed<<setprecision(6)<<e<<endl;}

     return 0;
}
