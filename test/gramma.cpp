#include<iostream>
#include <armadillo>
using namespace arma;
int main(){
    int dim1=13;
    int dim2=6;
    mat A=mat(dim1,dim2).randn();
    A=A*10*-1;
    A.print();
    double trans_array[dim2][dim1];
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            trans_array[i][j]=A.at(j,i);
        }
    }
    // check memory in arrays
    std::cout<<trans_array<<std::endl;
    std::cout<<*trans_array<<std::endl;
    std::cout<< (*(*(trans_array+1)+3))<<std::endl;
    
    arma::mat F= arma::mat((double *)trans_array ,dim1,dim2,false,true);
    std::cout<<"slice\n";
    F.submat(span(2,3),span(4,5)).print();
    std::cout<<"slice arr\n";
    arma::mat S= arma::mat((double *)(trans_array+1) ,dim1,2,false,true);
    S.print();
    std::cout<<"F\n";
    F.print();
    return 0;
}
