#include<iostream>
#include<string>
#include <algorithm>
#include <armadillo>
#include<vector>
#include<cmath>
#include<omp.h>
#include<chrono>
using namespace arma;
void house( double x[],int size , double v[],double & beta);
void serial_qr(double * arr,int column ,int row,double *betas);
void recover(double * arr,int column ,int row,const double *betas);
void block_qr(double * arr,int column ,int row,int r,double * q_arr);
int main(int argc, char const *argv[])
{
    // initial matrix
    std::string tile_num=argv[1];
    std::string col_num=argv[2];
    std::string row_num=argv[3];
    std::string thread_num=argv[3];
    int b=std::stoi(tile_num);
    int p=std::stoi(col_num);
    int q=std::stoi(row_num);
    int tn=std::stoi(thread_num);
    int dim1=b*p;
    int dim2=b*q;
    mat A=mat(dim1,dim2).randn();
    A=A*10*-1;
    double trans_array[dim2][dim1];
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            trans_array[i][j]=A.at(j,i);
        }
    }
    // A.print();
    /*  testing serial qr 
    double betas[dim2];
    serial_qr((double *) trans_array,dim1 ,dim2,betas);
    recover((double *)trans_array,dim1,dim2,betas);
    arma::mat F= arma::mat((double *)trans_array ,dim1,dim2,false,true);
    F.print();
    std::cout<<norm2est(F-A,2);
    */
    /*
    arma::mat G= arma::mat((double *) trans_array,dim1,dim2,false,true);
    double Q_array[dim1][dim1];
    block_qr((double *)trans_array,dim1,dim2,2,(double *)Q_array);
    arma::mat Q= arma::mat((double *) Q_array,dim1,dim1,false,true);
    // testing ,update array again.
    for (int i=0;i<dim2;i++){
        for (int j=0;j<dim1;j++){
            trans_array[i][j]=A.at(j,i);
        }
    }
     */
    // block parameter r
    int r=16;
    arma::mat T= arma::mat((double *) trans_array,dim1,dim2,false,true);
    auto start=std::chrono::high_resolution_clock::now();
    for (int k=0;k<q;k++){
    // tiled using four functions 
    //1.DGEQRT. diagonal
    // // copy digonal part in continous memory
        double A_kk_trans[b][b];
        double Q_arr[b][b];
        arma::mat A_kk= arma::mat((double *) A_kk_trans,b,b,false,true);
        A_kk=T.submat(span(k*b,(k+1)*b-1),span(k*b,(k+1)*b-1));
        // std::cout<<"Akk\n"<<A_kk;
        block_qr((double *) A_kk_trans,b,b, r,(double * )Q_arr);
        arma::mat Q_kk= arma::mat((double *) Q_arr,b,b,false,true);
        // update A
        T.submat(span(k*b,(k+1)*b-1),span(k*b,(k+1)*b-1))=A_kk;
        #pragma omp parallel for num_threads(tn)
        for (int j=k+1;j<q;j++){
            //2.DLARFB LEVEL 3 BLAS OPTIONS UPDATE LEFT 
            T.submat(span(k*b,(k+1)*b-1),span(j*b,(j+1)*b-1))=Q_kk.t()*
                                    T.submat(span(k*b,(k+1)*b-1),span(j*b,(j+1)*b-1));
            // std::cout<<"k is "<<k<<" j is "<<j<<"\n"<<T<<"\n";
            }

        //3. DTSQRT Coupling 
        for (int i =k+1;i<p;i++){
            // copy memory
            double A_ik_trans[b][2*b];
            double Q_ik_arr[2*b][2*b];
            arma::mat A_ik= arma::mat((double *) A_ik_trans,2*b,b,false,true);
            A_ik.rows(0,b-1)=T.submat(span(k*b,(k+1)*b-1),span(k*b,(k+1)*b-1));
            A_ik.rows(b,2*b-1)=T.submat(span(i*b,(i+1)*b-1),span(k*b,(k+1)*b-1));
            // update 
            block_qr((double *) A_ik_trans,2*b,b, r,(double * )Q_ik_arr);
            // copy back 
            T.submat(span(k*b,(k+1)*b-1),span(k*b,(k+1)*b-1))=A_ik.rows(0,b-1);
            T.submat(span(i*b,(i+1)*b-1),span(k*b,(k+1)*b-1))=A_ik.rows(b,2*b-1);
            arma::mat Q_ik= arma::mat((double *) Q_ik_arr,2*b,2*b,false,true);
            #pragma omp parallel for num_threads(tn)
            for (int s=k+1;s<q;s++){
                // 4. update coupling left
                // copy memory
                double A_is_trans[b][2*b];
                arma::mat A_is= arma::mat((double *) A_is_trans,2*b,b,false,true);
                A_is.rows(0,b-1)=T.submat(span(k*b,(k+1)*b-1),span(s*b,(s+1)*b-1));
                A_is.rows(b,2*b-1)=T.submat(span(i*b,(i+1)*b-1),span(s*b,(s+1)*b-1));
                //  update 
                A_is=Q_ik.t()*A_is;
                // copy back 
                T.submat(span(k*b,(k+1)*b-1),span(s*b,(s+1)*b-1))=A_is.rows(0,b-1);
                T.submat(span(i*b,(i+1)*b-1),span(s*b,(s+1)*b-1))=A_is.rows(b,2*b-1);
            }
        }
    }
    auto end=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"duration: "<<duration<<"ms"<<std::endl;
    // T.print();
    // std::cout<<T-G;
    // std::cout<<"?\n";
    //     std::cout<<"Even this is wrong?";
    // block_qr((double * )trans_array,dim1 ,dim2,3);
    // std::cout<<"results:\n";
    // recover((double *)trans_array,dim1,dim2);
    // arma::mat F= arma::mat((double *)trans_array ,dim1,dim2,false,true);
    // std::cout<<"Recover:\n";
    // F.print();
    // std::cout<<norm2est(F-A,2);
    return 0;
}
void house( double x[],int size , double v[],double & beta){
    arma::vec x_vec=arma::vec(x,size,false,false);
    arma::vec v_vec=arma::vec(v,size,false,false); // use the same memory of double v ;
    for (int i =0;i<size;i++){
        v_vec[i]=x_vec[i];
    }
    v_vec[0]=1.0;
    double sigma;
    if(size-1>=1) {
        sigma=dot(x_vec.subvec(1,size-1),x_vec.subvec(1,size-1));
    }else{
        sigma=0;
    }
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
void serial_qr(double * arr,int column ,int row,double *betas){
    bool test=true;
    // convert data from array to met
    int dim1=column;
    int dim2=row;
    // double betas[dim2];
    arma::mat A= arma::mat(arr ,column,row,false,true);
    // std::cout<<"In serial qr \n"<<A;
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
        if (dim1-iter_dim-1>0){
            A.col(iter_dim).tail(dim1-iter_dim-1)=v_vec.subvec(1,dim1-iter_dim-1);
        }        
    }
}
void recover(double * arr,int column ,int row,const double * betas){
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
                                            -betas[dim2-1-i]*v_t
                      *(v_t.t()*F.submat(span(dim2-1-i, dim1-1), span(dim2-1-i, dim2-1)));
    }
}
void block_qr(double * arr,int column ,int row,int r,double * q_arr){
    int begin_mark=0;
    int k=0;
    // for temp array size
    int temp_col;
    int temp_row;
    int dim1=column;
    int dim2=row;
    arma::mat T= arma::mat((double *) arr,dim1,dim2,false,true);
    arma::mat Q= arma::mat((double *) q_arr,dim1,dim1,false,true);
    Q=eye(dim1,dim1);
    // T.print();
    // use temp_array to do slice.
    while (begin_mark <dim2){
        int tail_mark =std::min(begin_mark+r-1,dim2-1);
        // using begin and tail to slice matrix.
        // trigangularize A(beign:m,begin:end)
        temp_col=tail_mark-begin_mark+1;
        temp_row=dim1-begin_mark;
        double * temp_array =new double[temp_col*temp_row];
        // std::cout<<"column is "<<temp_col <<"row is "<<temp_row<<"\n";
        for (int i=0;i<temp_col;i++){
            for (int j=0;j<temp_row;j++){
                *(temp_array+i*temp_row+j)=T.submat(span(begin_mark,dim1-1),span(begin_mark,tail_mark)).at(j,i);
            }
        }
        double betas[temp_col];
        serial_qr( temp_array,temp_row ,temp_col,betas);
        arma::mat S= arma::mat((double *) temp_array,temp_row,temp_col,true,true);
        T.submat(span(begin_mark,dim1-1),span(begin_mark,tail_mark))=S;        
        // generate block representation.
        mat W=mat(temp_row,temp_col).zeros();
        mat Y=mat(temp_row,temp_col).zeros();
        mat I_k=eye(temp_row,temp_row);
        for (int j =0;j<temp_col;j++){
            arma::vec v_vec_result=arma::vec(temp_row).zeros();
            v_vec_result[j]=1;
            v_vec_result.tail(temp_row-j-1)=S.col(j).tail(temp_row-j-1);
            // erase v vector in A matrix 
            T.col(begin_mark+j).tail(temp_row-j-1).zeros();
            double beta=betas[j];
            arma::vec z = beta*(I_k-W*Y.t())*v_vec_result;
            W.col(j)=z;
            Y.col(j)=v_vec_result;
        }
        delete [] temp_array;
        // update other A
        if (tail_mark<dim2-1){
        T.submat(span(begin_mark,dim1-1),span(tail_mark+1,dim2-1))=(I_k-W*Y.t()).t()
                         *T.submat(span(begin_mark,dim1-1),span(tail_mark+1,dim2-1));
        }
        // update Q
        Q.cols(begin_mark,dim1-1)=Q.cols(begin_mark,dim1-1)*(I_k-W*Y.t());
        begin_mark=tail_mark+1;
    }
}