#include<iostream>
#include <armadillo>
#include<vector>
#include<cmath>
using namespace arma;
void house( double x[],int size , double v[],double & beta);
int main()
{
    // initial matrix
    int dim1=10;
    int dim2=8;
    mat A=mat(dim1,dim2).randn();
    A=A*10*-1;
    A.print();
    // generate x v and beta 
    double x[dim1];
    double v[dim1];
    std::vector<double> v_std_vector;
    double beta;
    arma::vec v_vec=arma::vec(v,dim1,false,false); // share the same memory of v
    int iter_dim;
    for (iter_dim=0;iter_dim<dim2;iter_dim++){
        for (int i=iter_dim;i<dim1;i++){
            x[i-iter_dim]=A.at(i,iter_dim);
        }
        house(x,dim1-iter_dim,v,beta);
    // applay
        A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))=
                A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))
              -beta * v_vec.subvec(0,dim1-iter_dim-1)*v_vec.subvec(0,dim1-iter_dim-1).t()*A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1));
    }
    A.print();
    return 0;
}
void house( double x[],int size , double v[],double & beta){
    arma::vec x_vec=arma::vec(x,size,false,false);
    arma::vec v_vec=arma::vec(v,size,false,false); // use the same memory of double v ;
    std::cout<<x_vec<<std::endl;
    for (int i =0;i<size;i++){
        v_vec[i]=x_vec[i];
    }
    v_vec[0]=1.0;
    double sigma=dot(x_vec.subvec(1,size-1),x_vec.subvec(1,size-1));
    if (sigma==0 && x_vec[0]>0){
        beta=0;
    }else if(sigma==0 && x_vec[0]<0){
        beta=-2;
    }else{
        double mu=std::sqrt(sigma + x_vec[0]*x_vec[0]);
        if (x_vec[0]<0){
            v_vec[0]=x_vec[0]-mu;

        }else{
            v_vec[0]=-1*sigma /(x_vec[0]+mu);
        }
        beta=2*v_vec[0]*v_vec[0]/(sigma+v_vec[0]*v_vec[0]);
        v_vec=v_vec/v_vec[0];
    }
}