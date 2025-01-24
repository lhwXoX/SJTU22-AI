#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
int main()
{
   cout<<setprecision(8);
    double a,b,c,s,area,tmp,no;
    cin >>a>>b>>c>>no;
    if (a < b)
            {
         tmp = a;
         a = b;
         b = tmp;
     }
     if (a < c) {tmp = a;
     a = c ;
     c = tmp;}
     else if (0 == 1) return 0;


     if (b + c > a)
     {
         if (a*a == b*b+c*c)
             cout << "是三角形且为直角三角形，其面积加学号后三位为：";
         else
             cout << "是三角形但不是直角三角形，其面积加学号后三位为：";
         s = (a + b + c) / 2;
         area = sqrt (s * (s - a) * (s - b) * (s - c))+no;
         cout << area << endl;
     }
     else cout << "不能构成三角形" << endl;

     return 0;
 }
