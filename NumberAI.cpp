#pragma GCC optimize(2)
#include<iostream>
#include<cmath>
#include<algorithm>
#include<graphics.h>
using namespace std;
#define e 2.718281828
const int number=300;
const int train_time=250;
//        num cnt pixel
float ima[30][6000][784];
float learn_rate=0.01;
const int n=40;
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
inline float cross_entropy_loss(float predicted, float actual) {
    return -actual * log(predicted + 1e-9); // 加上一个小常数防止 log(0)
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
    float train() {
	    float loss_sum = 0;
	    for (int num = 0; num < 10; num++) {
	        for (int id = 1; id <= number; id++) {
	            run(num, id);  // 前向传播
	
	            // 计算交叉熵损失
	            float output_grad[10] = {0};
	            for (int i = 0; i < 10; i++) {
	                float predicted = Output[i].get_value();
	                float actual = (i == num) ? 1 : 0;
	                loss_sum += cross_entropy_loss(predicted, actual);
	                output_grad[i] = predicted - actual;  // 交叉熵损失的梯度
	            }
	
	            // 反向传播更新权重
	            for (int i = 0; i < 10; i++) {
	                for (int j = 0; j < n; j++) {
	                    Output[i].w[j] -= learn_rate * output_grad[i] * Hidden[1][j].get_value();
	                }
	                Output[i].b -= learn_rate * output_grad[i];
	            }
	
	            // 隐藏层2的梯度计算
	            float hidden2_grads[n] = {0};
	            for (int h2 = 0; h2 < n; h2++) {
	                float grad = 0;
	                for (int out = 0; out < 10; out++) {
	                    grad += Output[out].w[h2] * output_grad[out];
	                }
	                grad *= sigmoid_derivative(Hidden[1][h2].get_value());
	                hidden2_grads[h2] = grad;
	
	                // 更新隐藏层2的权重
	                for (int j = 0; j < n; j++) {
	                    Hidden[1][h2].w[j] -= learn_rate * grad * Hidden[0][j].get_value();
	                }
	                Hidden[1][h2].b -= learn_rate * grad;
	            }
	
	            // 隐藏层1的梯度计算
	            for (int h1 = 0; h1 < n; h1++) {
	                float grad = 0;
	                for (int h2 = 0; h2 < n; h2++) {
	                    grad += Hidden[1][h2].w[h1] * hidden2_grads[h2];
	                }
	                grad *= sigmoid_derivative(Hidden[0][h1].get_value());
	
	                // 更新隐藏层1的权重
	                for (int j = 0; j < 784; j++) {
	                    Hidden[0][h1].w[j] -= learn_rate * grad * Input[j].get_value();
	                }
	                Hidden[0][h1].b -= learn_rate * grad;
	            }
	        }
	    }
	    return loss_sum;
	}
};
AI ai;
int main(){
	initgraph(train_time,400);
	load_data();
	ai.init();
	cout<<endl<<"------------训练中"<<"------------" <<endl;
	int last=800;
	int last2=800;
	for(int i=1;i<train_time;i++){
		int loss=ai.train();
		setcolor(EGERGB(255,255,255));
		/*if(loss>last){
			setcolor(EGERGB(255,0,0));
			xyprintf(i,800-loss,"%d",i);
		}*/
		line(i,400-loss,i-1,400-last);
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
		line(i,400-loss2,i-1,400-last2);
		last2=loss2;
		printf("次数%d 训练集损失函数:%d 测试集损失函数:%d         \r",i,loss,(int)loss2);
	}
	cout<<endl<<"------------训练结束------------" <<endl;
	cout<<endl<<"------------结果------------" <<endl;
	int err_cnt1=0;
	for(int i=0;i<10;i++){
		cout<<"数字"<<i<<endl;
		cout<<"预测结果";
		for(int j=1;j<=number;j++){
			ai.run(i,j);
			float maxx=ai.Output[0].get_value();
			int res=0;
			for(int tmp=1;tmp<10;tmp++){
				if(ai.Output[tmp].get_value()>maxx){
					res=tmp;
					maxx=ai.Output[tmp].get_value();
				}
			}
			cout<<res<<" ";
			if(res!=i){
				err_cnt1++;
			}
		}
		cout<<endl;
	}
	cout<<"训练集准确率："<<100.0*(10*number-err_cnt1)/(10*number)<<"%"<<endl;
	int err_cnt2=0;
	for(int i=10;i<20;i++){
		cout<<"数字"<<i-10<<endl;
		cout<<"预测结果";
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
			if(res!=i-10){
				err_cnt2++;
			}
		}
		cout<<endl;
	}
	cout<<"测试集准确率："<<100.0*(10*10-err_cnt2)/(10*10)<<"%";
	getch();
} 
