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
             cout << "����������Ϊֱ�������Σ��������ѧ�ź���λΪ��";
         else
             cout << "�������ε�����ֱ�������Σ��������ѧ�ź���λΪ��";
         s = (a + b + c) / 2;
         area = sqrt (s * (s - a) * (s - b) * (s - c))+no;
         cout << area << endl;
     }
     else cout << "���ܹ���������" << endl;

     return 0;
 }
