#pragma GCC optimize(2)
#include<iostream>
#include<cmath>
#include<algorithm>
#include<graphics.h>
using namespace std;
#define e 2.718281828
const int number=300;
const int train_time=1250;
const int Smaller=20;
//        num cnt pixel
float ima[30][6000][784];
float learn_rate=0.01;
const int n=20;
#include"headfile/Dataloader.hpp"
inline float sigmoid(float x){
	return 1/(1+pow(e,-x));
}
inline float sigmoid_derivative(float x){
    return x * (1 - x);
}
inline float mse_loss(float predicted, float actual) {
    return 0.5 * (predicted - actual) * (predicted - actual);
}
class NODE{
	public:
	NODE * from[1000];
	float w[1000];
	float b;
	int cnt;
	float value;
		void band(NODE * node){
			from[cnt++]=node;
		}
		void init(){
			for(int i=0;i<cnt;i++) w[i]=1.0*(rand()%200-100)/100;
			b=1.0*(rand()%200-100)/100;
		}
		void run(){
			value=0;
			for(int i=0;i<cnt;i++){
				value+=from[i]->get_value()*w[i];
			}
			value+=b;
			value=sigmoid(value);
		}
		float get_value(){
			return value;
		}
		void set(float v){
			value=v;
		}
};
class AI{
public:
    NODE Input[784];
    NODE Hidden[2][n]; // 两个隐藏层，每个层20个节点
    NODE Output[10];
    void init(){
        // 初始化第一个隐藏层
        for(int i=0; i<n; i++){
            for(int j=0; j<784; j++){
                Hidden[0][i].band(&Input[j]);
            }
            Hidden[0][i].init();
        }
        // 初始化第二个隐藏层
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                Hidden[1][i].band(&Hidden[0][j]);
            }
            Hidden[1][i].init();
        }
        // 初始化输出层
        for(int i=0; i<10; i++){
            for(int j=0; j<n; j++){
                Output[i].band(&Hidden[1][j]);
            }
            Output[i].init();
        }
    }
    void run(int num, int id){
        for(int i=0; i<784; i++){
            Input[i].set(ima[num][id][i]);
        }
        for(int i=0; i<n; i++){
            Hidden[0][i].run();
        }
        for(int i=0; i<n; i++){
            Hidden[1][i].run();
        }
        for(int i=0; i<10; i++){
            Output[i].run();
        }
    }
    float train(){
    	float loss_sum=0;
        for (int num = 0; num < 10; num++) {
            for (int id = 1; id <= number; id++) {
                run(num, id);  // 运行网络
                for (int i = 0; i < 10; i++) {
                    register float predicted = Output[i].get_value();
                    register float actual = i == num ? 1 : 0;
                    register float loss = mse_loss(predicted, actual);
                    loss_sum+=loss;
                    // 反向传播      输出层
                    register float output_grad = predicted - actual;
                    for (int j = 0; j < n; j++) {
                        Output[i].w[j] -= learn_rate * output_grad * Hidden[1][j].get_value();
                    }
                    Output[i].b -= learn_rate * output_grad;
                    
                    // 反向传播 第二个隐藏层
                    register float hidden2_grad = 0;
                    for (int j = 0; j < 10; j++) {
                        hidden2_grad += Output[j].w[i] * (Output[j].get_value() - (j == num ? 1 : 0));
                    }
                    hidden2_grad *= sigmoid_derivative(Hidden[1][i].get_value()); // sigmoid函数导数
                    for (int j = 0; j < n; j++) {
                        Hidden[1][i].w[j] -= learn_rate * hidden2_grad * Hidden[0][j].get_value();
                    }
                    Hidden[1][i].b -= learn_rate * hidden2_grad;
                    
                    // 反向传播 第一个隐藏层
                    register float hidden1_grad = 0;
                    for (int j = 0; j < n; j++) {
                        hidden1_grad += Hidden[1][j].w[i] * hidden2_grad;
                    }
                    hidden1_grad *= sigmoid_derivative(Hidden[0][i].get_value()); // sigmoid函数导数
                    for (int j = 0; j < 784; j++) {
                        Hidden[0][i].w[j] -= learn_rate * hidden1_grad * Input[j].get_value();
                    }
                    Hidden[0][i].b -= learn_rate * hidden1_grad;
                }
            }
        }
        return loss_sum;
    }
};
AI ai;
int main(){
	initgraph(train_time,800);
	load_data();
	ai.init();
	for(int n=1;n<=1;n++){
		cout<<endl<<"------------训练中"<<n<<"------------" <<endl;
		int last=800;
		int last2=800;
		for(int i=1;i<train_time;i++){
			if(i==train_time/6) learn_rate/=Smaller;
			if(i==train_time/6*2) learn_rate/=Smaller;
			if(i==train_time/6*3) learn_rate/=Smaller;
			if(i==train_time/6*4) learn_rate/=Smaller;
			if(i==train_time/6*5) learn_rate/=Smaller;
			int loss=ai.train();
			setcolor(EGERGB(255,255,255));
			/*if(loss>last){
				setcolor(EGERGB(255,0,0));
				xyprintf(i,800-loss,"%d",i);
			}*/
			line(i,800-loss,i-1,800-last);
			last=loss;
			
			float loss2=0;
			for(int i=10;i<20;i++){
				for(int j=1;j<=10;j++){
					ai.run(i,j);
					for(int tmp=1;tmp<10;tmp++){
						loss2+=mse_loss(tmp==i-10?1:0,ai.Output[tmp].get_value());
					}
				}
			}
			loss2*=18;
			setcolor(EGERGB(0,255,0));
			if(loss2>last2) setcolor(EGERGB(255,0,0));
			line(i,800-loss2,i-1,800-last2);
			last2=loss2;
			printf("round:%d loss:%d val:%d         \r",i,loss,(int)loss2);
			
			//Sleep(1);
		}
		cout<<endl<<"------------训练结束------------" <<endl;
		cout<<endl<<"------------结果------------" <<endl;
		for(int i=0;i<10;i++){
			cout<<"number"<<i<<endl;
			cout<<"predict:";
			for(int j=1;j<=number;j++){
				ai.run(i,j);
				float maxx=ai.Output[0].get_value();
				int res=0;
				for(int tmp=1;tmp<10;tmp++){
					//printf("%.3f ",ai.Output[tmp].get_value());
					if(ai.Output[tmp].get_value()>maxx){
						res=tmp;
						maxx=ai.Output[tmp].get_value();
					}
				}
				//cout<<endl;
				cout<<res<<" ";
			}
			cout<<endl;
		}
		learn_rate/=2.0;
	}
	for(int i=10;i<20;i++){
		cout<<"number"<<i-10<<endl;
		cout<<"predict:";
		for(int j=1;j<=10;j++){
			ai.run(i,j);
			float maxx=ai.Output[0].get_value();
			int res=0;
			for(int tmp=1;tmp<10;tmp++){
				//printf("%.3f ",ai.Output[tmp].get_value());
				if(ai.Output[tmp].get_value()>maxx){
					res=tmp;
					maxx=ai.Output[tmp].get_value();
				}
			}
			//cout<<endl;
			cout<<res<<" ";
		}
		cout<<endl;
	}
	getch();
} 
