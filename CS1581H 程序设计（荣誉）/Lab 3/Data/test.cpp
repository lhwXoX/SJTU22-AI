#include <iostream>
using namespace std;
int main (){
    int n,seq;
    int num = 0;//num records the times of swap.
    cin >> n >> seq;
    //n is the length of the array.
    //If seq==1,then the sequence is in ascending order
	//If seq==0 then in descending order.
    int arr[n];
    for (int i = 0; i < n;++i)
        cin >> arr[i];
    
    bubble_sort(arr, n, seq,num);

    for (int i = 0; i < n;++i)
        cout << arr[i] << ' ';
    cout << endl << num;
}
void bubble_sort(int arr[], int n, bool seq,int num){
    bool flag = 0;
    //if no number is swapped,then break the loop.
    //we use the flag to record this.
    for (int i = 1; i < n;++i)
        for (int j = 0; j <= n;++j){
            if(arr[j]<arr[j+1]&&seq){
                swap(arr[j], arr[j + 1]);
                ++num;
                flag = 1;
            }
            else if(arr[j]>arr[j+1]&&seq){
                swap(arr[j], arr[j + 1]);
                ++num;
                flag = 1;
            }
        }
        if(!flag)   break;
    
}
