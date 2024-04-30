#include<iostream>
#include<string>
int main(int argc,const char* argv[])
{
    std::string str=argv[1];
    int num=std::stoi(str);
    std::cout<<num+100;
    return 0;
}