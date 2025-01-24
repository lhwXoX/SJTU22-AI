#include <iostream>
using namespace std;
void bubble_sort(int arr[], int n, bool seq,int * num);
int main ()
{
    int n,seq;
    int  num = 0;
    cin >> n >> seq;
    int arr[n];
    for (int i = 0; i < n;++i) cin >> arr[i];
    bubble_sort(arr, n, seq, &num);
    for (int i = 0; i < n;++i) cout << arr[i] << ' ';
    cout << endl << num;
}
void bubble_sort(int arr[], int n, bool seq,int * num)
{
    for (int i = 0; i < n;++i)
    {
        bool flag = 0;
        for (int j = 0; j < n - i;++j)
        {
            if(arr[j]<arr[j+1]&&!seq)
            {
                swap(arr[j], arr[j + 1]);
                ++*num;
                flag = 1;
            }
            else if(arr[j]>arr[j+1]&&seq)
            {
                swap(arr[j], arr[j + 1]);
                ++*num;
                flag = 1;
            }
        }
        if(!flag) break;
    }
}
