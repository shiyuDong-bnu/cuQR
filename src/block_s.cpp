#include<iostream>
#include <armadillo>
#include<vector>
#include<cmath>
using namespace arma;
void house( double x[],int size , double v[],double & beta);
void serial_qr(double * arr,int column ,int row,double * betas);
int main()
{
    // initial matrix
    int dim1=10;
    int dim2=8;
    mat A=mat(dim1,dim2).randn();
    A=A*10*-1;
    mat D=mat(A);
    A.print();
    // begin funcitons.
    double x[dim1];
    double v[dim1];
    double betas[dim2];
    std::vector<double> v_std_vector;
    arma::vec v_vec=arma::vec(v,dim1,false,false); // share the same memory of v
    int iter_dim;
    for (iter_dim=0;iter_dim<dim2;iter_dim++){
        for (int i=iter_dim;i<dim1;i++){
            x[i-iter_dim]=A.at(i,iter_dim);
        }
        house(x,dim1-iter_dim,v,betas[iter_dim]);
    // applay
        A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))=
                A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))
              -betas[iter_dim] * v_vec.subvec(0,dim1-iter_dim-1)*v_vec.subvec(0,dim1-iter_dim-1).t()*A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1));
    }
    A.print();
    // using funciton to repeat the process
    // initize data
    D.print();
    double trans_array[dim2][dim1];
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            trans_array[i][j]=D.at(j,i);
        }
    }
    arma::mat T= arma::mat((double *)trans_array ,dim1,dim2,false,true);
    serial_qr((double *)trans_array,dim1,dim2,betas);
    std::cout<<"Outside serial qr";
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            std::cout<<trans_array[i][j]<<" ";
        }
        std::cout<<"\n";
    }
    return 0;
}
void house( double x[],int size , double v[],double & beta){
    arma::vec x_vec=arma::vec(x,size,false,false);
    arma::vec v_vec=arma::vec(v,size,false,false); // use the same memory of double v ;
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
void serial_qr(double * arr,int column ,int row,double * betas){
    // convert data from array to met
    int dim1=column;
    int dim2=row;
    arma::mat A= arma::mat(arr ,column,row,false,true);
    std::cout<<"IN serial qr\n";
    A.print();
    arma::mat B=arma::mat(A).zeros();
    arma::mat Q=arma::eye(dim1,dim1);
    arma::mat D=arma::mat(A);
    double x[dim1];
    double v[dim1];
    std::vector<double> v_std_vector;
    arma::vec v_vec=arma::vec(v,dim1,false,false); // share the same memory of v
    int iter_dim;
    for (iter_dim=0;iter_dim<dim2;iter_dim++){
        for (int i=iter_dim;i<dim1;i++){
            x[i-iter_dim]=A.at(i,iter_dim);
        }
        house(x,dim1-iter_dim,v,betas[iter_dim]);
    // applay
        A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))=
                A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1))
              -betas[iter_dim] * v_vec.subvec(0,dim1-iter_dim-1)*v_vec.subvec(0,dim1-iter_dim-1).t()*A.submat(arma::span(iter_dim,dim1-1),arma::span(iter_dim,dim2-1));
        B.col(iter_dim).tail(dim1-iter_dim-1)=v_vec.subvec(1,dim1-iter_dim-1);
    }
    A.print();
    // reconstruct A
    B.print();
    // matrix Q
    for (int i=0;i<dim2;i++){
        std::cout<<"Q vector";
        std::cout<<Q;
        arma::vec v_vec_trans=arma::vec(v,dim1,true,false);// not share the same memory
        v_vec_trans.zeros();
        v_vec_trans[dim2-i-1]=1;
        v_vec_trans.tail(dim1-dim2+i)=B.col(dim2-1-i).tail(dim1-dim2+i);
        std::cout<<"\n";
        std::cout<<v_vec_trans<<std::endl;
        std::cout<<betas[dim2-i]<<std::endl;
        std::cout<<dot(v_vec_trans,v_vec_trans)<<std::endl;
        Q=Q-2/dot(v_vec_trans,v_vec_trans)*v_vec_trans*(v_vec_trans.t()*Q);
    }
    std::cout<<"Q matrix:\n";
    std::cout<<Q;
    std::cout<<" Q*R -A MATRIX \n";
    std::cout<<Q*A-D;
}