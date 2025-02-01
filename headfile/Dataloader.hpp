
void load_data(){
	cout<<"------------加载中------------" <<endl<<endl;
	for(int i=0;i<10;i++){
		for(int j=1;j<=number;j++){
			PIMAGE p=newimage();
			string dir="";
			dir="mnist_images\\";
			dir+=char(i+'0');
			dir+="\\";
			dir+=to_string(j);
			dir+=".png";
			cout<<"finish:"<<dir<<endl;
			getimage(p,dir.c_str());
			int cnt=0;
			for(int y=0;y<28;y++){
				for(int x=0;x<28;x++){
					color_t color=getpixel(x,y,p);
					int value=EGEGET_R(color)+EGEGET_G(color)+EGEGET_B(color);
					value/=3;
					ima[i][j][cnt++]=1.0*value/255.0;
				}
			}
			delimage(p);
		}
		
		for(int j=1;j<=10;j++){
			PIMAGE p=newimage();
			string dir="";
			dir="mnist_images\\";
			dir+=char(i+'0');
			dir+="\\";
			dir+=to_string(j+number+20);
			dir+=".png";
			cout<<"finish:"<<dir<<endl;
			getimage(p,dir.c_str());
			int cnt=0;
			for(int y=0;y<28;y++){
				for(int x=0;x<28;x++){
					color_t color=getpixel(x,y,p);
					int value=EGEGET_R(color)+EGEGET_G(color)+EGEGET_B(color);
					value/=3;
					ima[i+10][j][cnt++]=1.0*value/255.0;
				}
			}
			delimage(p);
		}
	}
	
	
	cout<<endl<<"------------加载完毕------------" <<endl;
	return;
}
