#include<iostream>
#include <armadillo>
#include<vector>
#include<cmath>
using namespace arma;
void house( double x[],int size , double v[],double & beta);
void serial_qr(double * arr,int column ,int row);
void recover(double * arr,int column ,int row);
int main()
{
    // initial matrix
    int dim1=13;
    int dim2=12;
    mat A=mat(dim1,dim2).randn();
    A=A*10*-1;
    double trans_array[dim2][dim1];
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            trans_array[i][j]=A.at(j,i);
        }
    }
    A.print();
    serial_qr((double *)trans_array,dim1,dim2);
    recover((double *)trans_array,dim1,dim2);
    arma::mat F= arma::mat((double *)trans_array ,dim1,dim2,false,true);
    F.print();
    std::cout<<norm2est(F-A,2);
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
void serial_qr(double * arr,int column ,int row){
    bool test=true;
    // convert data from array to met
    int dim1=column;
    int dim2=row;
    double betas[dim2];
    arma::mat A= arma::mat(arr ,column,row,false,true);
    double x[dim1];
    double v[dim1];
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
        A.col(iter_dim).tail(dim1-iter_dim-1)=v_vec.subvec(1,dim1-iter_dim-1);
    }
}
void recover(double * arr,int column ,int row){
    int dim1=column;
    int dim2=row;
    arma::mat F= arma::mat((double *)arr ,dim1,dim2,false,true);
    // recover A
    arma::vec v_vec_result=arma::vec(dim1).zeros();
    for (int i=0;i<dim2;i++){
        v_vec_result[dim2-i-1]=1;
        v_vec_result.tail(dim1-dim2+i)=F.col(dim2-1-i).tail(dim1-dim2+i);
        arma::vec v_t=v_vec_result.subvec(dim2-i-1,dim1-1);
        // construct R
        F.col(dim2-1-i).tail(dim1-dim2+i).zeros();
        F.submat(span(dim2-1-i, dim1-1), span(dim2-1-i, dim2-1))=F.submat(span(dim2-1-i, dim1-1), span(dim2-1-i, dim2-1))
                                            -2/dot(v_t,v_t)*v_t
                      *(v_t.t()*F.submat(span(dim2-1-i, dim1-1), span(dim2-1-i, dim2-1)));
    }
}